import os, time, json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
import httpx
import tiktoken
import uuid

# Updated imports for new structure
from app.services import conversation
from app.services import prompt_builder
from app.services.momentum import generate_momentum
from app.utils.debug import DebugTracer
from app.config import OPENAI_API_KEY, FT_MODEL, BASE_MODEL, PORT

# Initialize database
conversation.init_db()

# ---- Structured Debug Events ----
class DebugBus:
    def __init__(self):
        self._turn_buffers = {}  # turn_id -> [events]

    def new_turn(self) -> str:
        tid = f"turn_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        self._turn_buffers[tid] = []
        return tid

    def emit(self, turn_id: str, step: str, label: str, data: dict | None = None):
        evt = {
            "ts": int(time.time()*1000),
            "turn_id": turn_id,
            "step": step,          # e.g., "build_card", "call_openai", "first_token"
            "label": label,        # short human text
            "data": data or {}
        }
        # console (one line per event)
        print(json.dumps({"evt":"dbg", **evt}))
        # buffer for UI
        self._turn_buffers.setdefault(turn_id, []).append(evt)

    def dump(self, turn_id: str) -> list[dict]:
        return list(self._turn_buffers.get(turn_id, []))

    def clear(self, turn_id: str):
        self._turn_buffers.pop(turn_id, None)

debugbus = DebugBus()

# naive in-memory chat history for this process
# THREADS[thread_id] = [{"role":"user"|"assistant", "content": "..."}]
THREADS: dict[str, list[dict]] = {}

if not OPENAI_API_KEY:
    print("[BOOT] WARNING: OPENAI_API_KEY is not set")
if not FT_MODEL:
    print("[BOOT] WARNING: FT_MODEL is not set (will fallback to BASE_MODEL)")

# Get the backend directory (parent of app)
BASE_DIR = Path(__file__).parent.parent

# --- Live settings (process-local; no restart required) ---
class LiveSettings(BaseModel):
    temperature: float = Field(0.5, ge=0.0, le=2.0)
    max_tokens: int = Field(120, ge=1, le=4096)
    history_limit: int = Field(6, ge=0, le=50)
    use_full_history: bool = False
    frequency_penalty: float = Field(0.3, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.1, ge=-2.0, le=2.0)

LIVE = LiveSettings()

def _apply_live_settings_to_stream_kwargs(kwargs: dict):
    # override generation knobs
    kwargs["temperature"] = LIVE.temperature
    kwargs["max_tokens"] = LIVE.max_tokens
    kwargs["frequency_penalty"] = LIVE.frequency_penalty
    kwargs["presence_penalty"] = LIVE.presence_penalty
    # nothing to add for history here (handled when building messages)
    return kwargs

# ---------- Shared HTTP Client ----------
_http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(30.0, connect=5.0),
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    transport=httpx.AsyncHTTPTransport(retries=1),
    headers={"Connection": "keep-alive"},
)

# ---------- App ----------
app = FastAPI(title="Ember MVP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "https://ember-backend-8ja1.onrender.com",
        "*"  # Keep this for now, remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html from frontend directory
@app.get("/", include_in_schema=False)
def root():
    # Frontend is now in ../frontend relative to backend directory
    path = BASE_DIR.parent / "frontend" / "index.html"
    if not path.exists():
        return JSONResponse(status_code=404, content={"detail": f"index.html not found at {path}"})
    return FileResponse(path)

@app.get("/login.html", include_in_schema=False)
def login_page():
    """Serve login page"""
    path = BASE_DIR.parent / "frontend" / "login.html"
    if not path.exists():
        return JSONResponse(status_code=404, content={"detail": f"login.html not found at {path}"})
    return FileResponse(path)

# Optional: serve any static assets (css/js) off /static/*
# Mount the frontend directory as static files
try:
    frontend_path = BASE_DIR.parent / "frontend"
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
except Exception as e:
    print(f"[BOOT] Warning: Could not mount static files: {e}")

# ---------- Types ----------
class ChatIn(BaseModel):
    text: str
    thread_id: Optional[str] = None
    user_id: Optional[str] = "ui_user"

class ChatOut(BaseModel):
    reply: str
    debug: Dict[str, Any]

# ---------- Token estimate ----------
def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for m in messages:
            total += len(enc.encode(m.get("content", ""))) + 4
        return total
    except Exception:
        return sum(max(1, len(m.get("content","")) // 4) for m in messages)

# ---------- OpenAI (non-stream) ----------
async def call_openai(messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "stream": False}
    payload = {**payload, **_apply_live_settings_to_stream_kwargs({})}
    t0 = time.perf_counter()
    resp = await _http_client.post(url, headers=headers, json=payload)
    dt_ms = int((time.perf_counter() - t0) * 1000)
    try:
        data = resp.json()
    except Exception:
        data = {"raw_text": await resp.aread()}
    return {"status_code": resp.status_code, "ms": dt_ms, "data": data}

# ---------- Rate Limiting ----------
async def check_rate_limit(user_id: str, limit: int = 50) -> tuple[bool, int]:
    """
    Check if user is within daily message limit.
    Returns (is_allowed, messages_used_today)
    """
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # Count messages from today
    from app.database.supabase_client import supabase
    result = supabase.table("messages") \
        .select("id", count="exact") \
        .eq("user_id", user_id) \
        .eq("role", "user") \
        .gte("created_at", today_start.isoformat()) \
        .execute()

    count = result.count or 0
    return (count < limit, count)

# ---------- Non-stream endpoint ----------
@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn, request: Request):
    # Load recent turns to build the MEMORY card
    user_id   = body.user_id or "ui_user"
    thread_id = body.thread_id or "default"

    # Rate limit check
    is_allowed, messages_today = await check_rate_limit(user_id, limit=50)
    if not is_allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Daily limit of 50 messages reached. Used: {messages_today}",
                "messages_today": messages_today,
                "limit": 50
            }
        )

    tracer = DebugTracer(user_id=user_id, thread_id=thread_id)

    try:
        # Create debug bus turn
        turn_id = debugbus.new_turn()
        debugbus.emit(turn_id, "start", "turn_begin", {"user_text": body.text[:120]})

        tracer.event("request:received", {"text_preview": body.text[:120]})

        # Build prompt card with live settings
        with tracer.span("build_card"):
            limit = LIVE.history_limit if not LIVE.use_full_history else 50
            recent = await conversation.get_recent(user_id, thread_id, limit=limit)

            debugbus.emit(turn_id, "build", "building_card", {
                "history_n": len(recent),
                "use_full_history": LIVE.use_full_history,
                "limit": limit
            })

            system_content = prompt_builder.build_system_with_card(recent)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": body.text}
            ]
            tracer.attach("system_prompt_preview", system_content[:800])
            tracer.attach("messages_preview", messages)

        approx_tokens = estimate_tokens(messages)
        model = FT_MODEL or BASE_MODEL

        # Log prompt ready
        debugbus.emit(turn_id, "prompt", "prompt_ready", {
            "system_len": len(system_content),
            "messages_roles": [m["role"] for m in messages],
            "est_tokens": approx_tokens
        })

        with tracer.span("openai:call", {"model": model}):
            result = await call_openai(messages, model=model)

        reply, finish_reason, openai_error = "", None, None
        if result["status_code"] == 200:
            try:
                choice = result["data"]["choices"][0]
                reply = choice["message"]["content"]
                finish_reason = choice.get("finish_reason")

                # persist user → assistant turns
                try:
                    await conversation.save_turn(user_id, thread_id, "user", body.text)
                    await conversation.save_turn(user_id, thread_id, "assistant", reply)
                except Exception as _e:
                    print("[STORE] failed to save turn:", _e)
            except Exception as e:
                openai_error = f"parse_error: {e}"
        else:
            openai_error = result["data"]

        # Emit completion event
        debugbus.emit(turn_id, "complete", "turn_end", {
            "total_ms": result["ms"],
            "chars": len(reply) if reply else 0,
            "status": "ok" if not openai_error else "error"
        })

        tracer.attach("final_reply", reply[:800])

        # Create prompt payload for inspection
        system_msg = next((m for m in messages if m.get("role") == "system"), None)
        prompt_payload = {
            "system": system_msg["content"] if system_msg else "",
            "messages": messages,  # exact order sent to OpenAI
            "digest": {
                "roles": [m["role"] for m in messages],
                "system_len": len(system_msg["content"]) if system_msg else 0,
                "msg_count": len(messages)
            }
        }

        debug = {
            "model": model, "thread_id": body.thread_id, "user_id": body.user_id,
            "request_tokens_est": approx_tokens, "http_ms": result["ms"],
            "status_code": result["status_code"], "finish_reason": finish_reason,
            "messages": messages, "raw_error": openai_error,
            "recent_count": len(recent),
            "system_preview": system_content[:2000],
            "live_settings": LIVE.model_dump(),
            "trace": tracer.to_payload(),
            "prompt": prompt_payload,
            "debug_events": debugbus.dump(turn_id),
        }

        debugbus.clear(turn_id)
        tracer.finalize(status="ok" if not openai_error else "error",
                       error=openai_error,
                       extra={"http_ms": result["ms"], "tokens_est": approx_tokens})

    except Exception as e:
        tracer.event("pipeline:error", {"error": str(e)[:300]}, level="error")
        debugbus.emit(turn_id, "error", "pipeline_error", {"error": str(e)[:300]})
        debugbus.clear(turn_id)
        tracer.finalize(status="error", error=str(e)[:300])
        raise

    print(json.dumps({
        "evt": "chat", "http_ms": result["ms"], "status": result["status_code"],
        "tokens_est": approx_tokens, "finish": finish_reason, "thread": body.thread_id,
    }))

    if openai_error and not reply:
        return JSONResponse(status_code=500, content={"reply": "(error)", "debug": debug})

    return {"reply": reply, "debug": debug}

# ---------- Stream endpoint (SSE over POST) ----------
@app.post("/chat/stream")
async def chat_stream(body: ChatIn):
    """
    Streams tokens via Server-Sent Events-like frames over POST.
    Each frame is a line starting with 'data: ' and ending with a blank line.
    """
    # Load recent turns to build the MEMORY card
    user_id   = body.user_id or "ui_user"
    thread_id = body.thread_id or "default"

    # Rate limit check
    is_allowed, messages_today = await check_rate_limit(user_id, limit=50)
    if not is_allowed:
        async def error_gen():
            yield f'data: {json.dumps({"type":"error","error":"Daily limit of 50 messages reached","messages_today":messages_today,"limit":50})}\n\n'
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # init thread history if needed and append current user turn
    hist = THREADS.setdefault(thread_id, [])
    hist.append({"role": "user", "content": body.text})

    tracer = DebugTracer(user_id=user_id, thread_id=thread_id)

    async def event_gen():
        try:
            # Create debug bus turn
            turn_id = debugbus.new_turn()
            debugbus.emit(turn_id, "start", "turn_begin", {"user_text": body.text[:120]})

            tracer.event("request:received", {"text_preview": body.text[:120]})

            # 1) Build prompt card
            with tracer.span("build_card"):
                # Respect live settings for history limit
                limit = LIVE.history_limit if not LIVE.use_full_history else 50
                recent = await conversation.get_recent(user_id, thread_id, limit=limit)

                debugbus.emit(turn_id, "build", "building_card", {
                    "history_n": len(recent),
                    "use_full_history": LIVE.use_full_history,
                    "limit": limit
                })

                system_content = prompt_builder.build_system_with_card(recent)
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": body.text}
                ]
                tracer.attach("system_prompt_preview", system_content[:800])
                tracer.attach("messages_preview", messages)

            model = FT_MODEL or BASE_MODEL

            # Log prompt ready
            debugbus.emit(turn_id, "prompt", "prompt_ready", {
                "system_len": len(system_content),
                "messages_roles": [m["role"] for m in messages],
                "est_tokens": estimate_tokens(messages)
            })

            # 2) Emit 'start' event with compact debug
            start_debug = {
                "trace": {
                    "trace_id": tracer.trace_id,
                    "turn_id": tracer.turn_id,
                    "user_id": user_id,
                    "thread_id": thread_id,
                },
                "prompt_stats": {
                    "system_chars": len(system_content),
                    "messages": len(messages),
                },
                "live_settings": LIVE.model_dump(),
            }
            yield f'data: {json.dumps({"type":"start","turn_id":turn_id,"model":model,"debug":start_debug,"debug_events":debugbus.dump(turn_id)})}\n\n'

            # 3) Call OpenAI (stream)
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": messages, "stream": True}
            payload = {**payload, **_apply_live_settings_to_stream_kwargs({})}

            full_reply = ""
            first_token_ms = None
            t0 = time.perf_counter()

            with tracer.span("openai:stream", {"model": model}):
                async with _http_client.stream("POST", "https://api.openai.com/v1/chat/completions",
                                               headers=headers, json=payload) as r:
                        async for line in r.aiter_lines():
                            if not line:
                                continue
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                obj = json.loads(data)
                                delta = obj["choices"][0]["delta"].get("content")
                                if delta:
                                    if first_token_ms is None:
                                        first_token_ms = int((time.perf_counter() - t0) * 1000)
                                        tracer.event("openai:first_token", {"ms": first_token_ms})
                                        debugbus.emit(turn_id, "latency", "first_token", {"ms": first_token_ms})
                                    full_reply += delta
                                    yield f'data: {json.dumps({"type":"content","chunk":delta})}\n\n'
                            except Exception:
                                # surface parse hiccups as debug frames
                                yield f'data: {json.dumps({"type":"debug","raw":line[:200]})}\n\n'

            total_ms = int((time.perf_counter() - t0) * 1000)

            # 4) Persist and finalize
            if full_reply:
                try:
                    await conversation.save_turn(user_id, thread_id, "user", body.text)
                    await conversation.save_turn(user_id, thread_id, "assistant", full_reply)
                except Exception as _e:
                    print("[STORE] failed to save streaming turn:", _e)

            # append assistant's full reply to history and call momentum
            hist.append({"role": "assistant", "content": full_reply})

            # call momentum on the last N turns
            try:
                mom = generate_momentum(
                    history=hist,                   # pass the whole thing (small at MVP)
                    current_user_text=body.text,   # the message that triggered this turn
                    max_turns=6,                    # keep it modest
                    temperature=0.8,                # slightly higher for creativity
                    max_tokens=220,
                    api_key=OPENAI_API_KEY
                )
                momentum_block = {
                    "ideas": mom.get("ideas", []),
                    "debug": mom.get("debug", {}),
                }
            except Exception as e:
                momentum_block = {"ideas": [], "debug": {"error": str(e)[:200]}}

            # Emit final debug bus event
            debugbus.emit(turn_id, "complete", "turn_end", {
                "total_ms": total_ms,
                "chars": len(full_reply)
            })

            # Attach final artifacts & emit 'complete'
            tracer.attach("final_text_preview", full_reply[:800])
            meta = {"timing_ms": {"first_token": first_token_ms, "total": total_ms}}

            # Create prompt payload for inspection
            system_msg = next((m for m in messages if m.get("role") == "system"), None)
            prompt_payload = {
                "system": system_msg["content"] if system_msg else "",
                "messages": messages,  # exact order sent to OpenAI
                "digest": {
                    "roles": [m["role"] for m in messages],
                    "system_len": len(system_msg["content"]) if system_msg else 0,
                    "msg_count": len(messages)
                }
            }

            # Full debug payload (trace + attachments + steps)
            full_debug = tracer.to_payload()
            full_debug["momentum"] = momentum_block
            yield f'data: {json.dumps({"type":"complete","turn_id":turn_id,"final_text":full_reply,"meta":meta,"messages":messages,"prompt":prompt_payload,"debug":full_debug,"debug_events":debugbus.dump(turn_id)})}\n\n'

            debugbus.clear(turn_id)
            tracer.finalize(status="ok", extra=meta)

        except Exception as e:
            err = str(e)[:300]
            tracer.event("pipeline:error", {"error": err}, level="error")
            debugbus.emit(turn_id, "error", "pipeline_error", {"error": err})
            yield f'data: {json.dumps({"type":"error","error":err,"debug":tracer.to_payload(),"debug_events":debugbus.dump(turn_id)})}\n\n'
            debugbus.clear(turn_id)
            tracer.finalize(status="error", error=err)

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/settings")
async def get_settings():
    return LIVE.model_dump()

class SettingsIn(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    history_limit: Optional[int] = None
    use_full_history: Optional[bool] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

@app.post("/settings")
async def update_settings(body: SettingsIn):
    global LIVE
    # Update only provided fields with validation
    data = LIVE.model_dump()
    for k, v in body.model_dump(exclude_none=True).items():
        data[k] = v
    try:
        updated = LiveSettings(**data)
    except ValidationError as e:
        return JSONResponse(status_code=400, content={"error": "invalid_settings", "details": e.errors()})
    # commit
    LIVE = updated
    return LIVE.model_dump()

@app.get("/metrics")
async def metrics():
    return {"ok": True, "model": FT_MODEL or BASE_MODEL}

if __name__ == "__main__":
    import uvicorn
    # For production, pass app object directly
    # For dev with reload, use "app.main:app" string
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        http="httptools"
    )
