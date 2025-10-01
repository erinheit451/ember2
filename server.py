import os
import time
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
import httpx
import tiktoken

# Local imports
import conversation_store as conv
import prompt_card as card
from debug import DebugTracer
from momentum import generate_momentum

# ---------- Configuration ----------
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FT_MODEL = os.getenv("FT_MODEL", "")
BASE_MODEL = os.getenv("BASE_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "8000"))

if not OPENAI_API_KEY:
    print("[BOOT] WARNING: OPENAI_API_KEY is not set")
if not FT_MODEL:
    print("[BOOT] INFO: FT_MODEL not set, using BASE_MODEL")

BASE_DIR = Path.cwd()

# ---------- Live Settings ----------
class LiveSettings(BaseModel):
    temperature: float = Field(0.5, ge=0.0, le=2.0)
    max_tokens: int = Field(120, ge=1, le=4096)
    history_limit: int = Field(6, ge=0, le=50)
    use_full_history: bool = False
    frequency_penalty: float = Field(0.3, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.1, ge=-2.0, le=2.0)

LIVE = LiveSettings()

def apply_live_settings(kwargs: dict) -> dict:
    """Apply live settings to OpenAI API kwargs"""
    kwargs.update({
        "temperature": LIVE.temperature,
        "max_tokens": LIVE.max_tokens,
        "frequency_penalty": LIVE.frequency_penalty,
        "presence_penalty": LIVE.presence_penalty
    })
    return kwargs

# ---------- Debug Bus ----------
class DebugBus:
    """Structured debug event collection per turn"""
    def __init__(self):
        self._turn_buffers: Dict[str, List[Dict]] = {}

    def new_turn(self) -> str:
        tid = f"turn_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        self._turn_buffers[tid] = []
        return tid

    def emit(self, turn_id: str, step: str, label: str, data: Optional[dict] = None):
        evt = {
            "ts": int(time.time() * 1000),
            "turn_id": turn_id,
            "step": step,
            "label": label,
            "data": data or {}
        }
        print(json.dumps({"evt": "dbg", **evt}))
        self._turn_buffers.setdefault(turn_id, []).append(evt)

    def dump(self, turn_id: str) -> List[dict]:
        return list(self._turn_buffers.get(turn_id, []))

    def clear(self, turn_id: str):
        self._turn_buffers.pop(turn_id, None)

debugbus = DebugBus()

# ---------- In-Memory Thread History ----------
# For momentum generation - persists within process
THREADS: Dict[str, List[Dict]] = {}

# ---------- HTTP Client ----------
_http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(30.0, connect=5.0),
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    transport=httpx.AsyncHTTPTransport(retries=1),
    headers={"Connection": "keep-alive"},
)

# ---------- FastAPI App ----------
app = FastAPI(title="Ember", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Routes ----------
@app.get("/", include_in_schema=False)
def root():
    """Serve frontend HTML"""
    path = BASE_DIR / "index.html"
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": f"index.html not found at {path}"}
        )
    return FileResponse(path)

# app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

# ---------- Models ----------
class ChatIn(BaseModel):
    text: str
    thread_id: Optional[str] = None
    user_id: Optional[str] = None

class ChatOut(BaseModel):
    reply: str
    debug: Dict[str, Any]

class SettingsIn(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    history_limit: Optional[int] = None
    use_full_history: Optional[bool] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

# ---------- Utilities ----------
def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    """Estimate token count for messages"""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for m in messages:
            total += len(enc.encode(m.get("content", ""))) + 4
        return total
    except Exception:
        return sum(max(1, len(m.get("content", "")) // 4) for m in messages)

async def call_openai_non_stream(
    messages: List[Dict[str, str]],
    model: str
) -> Dict[str, Any]:
    """Make non-streaming OpenAI API call"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = apply_live_settings({
        "model": model,
        "messages": messages,
        "stream": False
    })
    
    t0 = time.perf_counter()
    resp = await _http_client.post(url, headers=headers, json=payload)
    dt_ms = int((time.perf_counter() - t0) * 1000)
    
    try:
        data = resp.json()
    except Exception:
        data = {"raw_text": await resp.aread()}
    
    return {
        "status_code": resp.status_code,
        "ms": dt_ms,
        "data": data
    }

def build_prompt_payload(messages: List[Dict]) -> Dict:
    """Create prompt inspection payload"""
    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    return {
        "system": system_msg["content"] if system_msg else "",
        "messages": messages,
        "digest": {
            "roles": [m["role"] for m in messages],
            "system_len": len(system_msg["content"]) if system_msg else 0,
            "msg_count": len(messages)
        }
    }

def ensure_valid_ids(user_id: Optional[str], thread_id: Optional[str]) -> tuple[str, str]:
    """
    Ensure user_id and thread_id are valid UUIDs.
    If not provided, generate new ones.
    If provided but invalid format, try to use them anyway (Supabase will validate).
    """
    if not user_id:
        user_id = str(uuid.uuid4())
    
    if not thread_id:
        thread_id = str(uuid.uuid4())
    
    return user_id, thread_id

# ---------- Chat Endpoints ----------
@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn, request: Request):
    """Non-streaming chat endpoint"""
    # Ensure valid UUIDs
    user_id, thread_id = ensure_valid_ids(body.user_id, body.thread_id)
    
    tracer = DebugTracer(user_id=user_id, thread_id=thread_id)
    turn_id = debugbus.new_turn()
    
    try:
        debugbus.emit(turn_id, "start", "turn_begin", {"user_text": body.text[:120]})
        tracer.event("request:received", {"text_preview": body.text[:120]})
        
        # Build prompt
        with tracer.span("build_card"):
            limit = LIVE.history_limit if not LIVE.use_full_history else 50
            
            try:
                recent = await conv.get_recent(user_id, thread_id, limit=limit)
            except Exception as e:
                # If conversation doesn't exist yet, start with empty history
                print(f"[INFO] No existing conversation, starting fresh: {e}")
                recent = []
            
            debugbus.emit(turn_id, "build", "building_card", {
                "history_n": len(recent),
                "use_full_history": LIVE.use_full_history,
                "limit": limit
            })
            
            system_content = card.build_system_with_card(recent)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": body.text}
            ]
            tracer.attach("system_prompt_preview", system_content[:800])
            tracer.attach("messages_preview", messages)
        
        approx_tokens = estimate_tokens(messages)
        model = FT_MODEL or BASE_MODEL
        
        debugbus.emit(turn_id, "prompt", "prompt_ready", {
            "system_len": len(system_content),
            "messages_roles": [m["role"] for m in messages],
            "est_tokens": approx_tokens
        })
        
        # Call OpenAI
        with tracer.span("openai:call", {"model": model}):
            result = await call_openai_non_stream(messages, model=model)
        
        reply, finish_reason, openai_error = "", None, None
        
        if result["status_code"] == 200:
            try:
                choice = result["data"]["choices"][0]
                reply = choice["message"]["content"]
                finish_reason = choice.get("finish_reason")
                
                # Persist to Supabase
                try:
                    await conv.save_turn(user_id, thread_id, "user", body.text)
                    await conv.save_turn(user_id, thread_id, "assistant", reply)
                except Exception as save_err:
                    print(f"[STORE] Failed to save turn: {save_err}")
                    # Don't fail the request if save fails - still return the reply
            except Exception as e:
                openai_error = f"parse_error: {e}"
                print(f"[OPENAI] Failed to parse response: {e}")
        else:
            openai_error = result["data"]
        
        # Debug events
        debugbus.emit(turn_id, "complete", "turn_end", {
            "total_ms": result["ms"],
            "chars": len(reply) if reply else 0,
            "status": "ok" if not openai_error else "error"
        })
        
        tracer.attach("final_reply", reply[:800])
        
        # Build debug payload
        debug = {
            "model": model,
            "thread_id": thread_id,
            "user_id": user_id,
            "request_tokens_est": approx_tokens,
            "http_ms": result["ms"],
            "status_code": result["status_code"],
            "finish_reason": finish_reason,
            "messages": messages,
            "raw_error": openai_error,
            "recent_count": len(recent),
            "system_preview": system_content[:2000],
            "live_settings": LIVE.model_dump(),
            "trace": tracer.to_payload(),
            "prompt": build_prompt_payload(messages),
            "debug_events": debugbus.dump(turn_id),
        }
        
        debugbus.clear(turn_id)
        tracer.finalize(
            status="ok" if not openai_error else "error",
            error=openai_error,
            extra={"http_ms": result["ms"], "tokens_est": approx_tokens}
        )
        
        print(json.dumps({
            "evt": "chat",
            "http_ms": result["ms"],
            "status": result["status_code"],
            "tokens_est": approx_tokens,
            "finish": finish_reason,
            "thread": thread_id,
        }))
        
        if openai_error and not reply:
            return JSONResponse(
                status_code=500,
                content={"reply": "(error)", "debug": debug}
            )
        
        return {"reply": reply, "debug": debug}
    
    except Exception as e:
        error_msg = str(e)[:300]
        tracer.event("pipeline:error", {"error": error_msg}, level="error")
        debugbus.emit(turn_id, "error", "pipeline_error", {"error": error_msg})
        debugbus.clear(turn_id)
        tracer.finalize(status="error", error=error_msg)
        
        # Return error response instead of raising
        return JSONResponse(
            status_code=500,
            content={"reply": f"Error: {error_msg}", "debug": {"error": error_msg}}
        )

@app.post("/chat/stream")
async def chat_stream(body: ChatIn):
    """Streaming chat endpoint with SSE"""
    # Ensure valid UUIDs
    user_id, thread_id = ensure_valid_ids(body.user_id, body.thread_id)
    
    # Maintain in-memory history for momentum
    hist = THREADS.setdefault(thread_id, [])
    hist.append({"role": "user", "content": body.text})
    
    tracer = DebugTracer(user_id=user_id, thread_id=thread_id)
    
    async def event_gen():
        turn_id = debugbus.new_turn()
        
        try:
            debugbus.emit(turn_id, "start", "turn_begin", {"user_text": body.text[:120]})
            tracer.event("request:received", {"text_preview": body.text[:120]})
            
            # Build prompt
            with tracer.span("build_card"):
                limit = LIVE.history_limit if not LIVE.use_full_history else 50
                
                try:
                    recent = await conv.get_recent(user_id, thread_id, limit=limit)
                except Exception as e:
                    # If conversation doesn't exist yet, start with empty history
                    print(f"[INFO] No existing conversation, starting fresh: {e}")
                    recent = []
                
                debugbus.emit(turn_id, "build", "building_card", {
                    "history_n": len(recent),
                    "use_full_history": LIVE.use_full_history,
                    "limit": limit
                })
                
                system_content = card.build_system_with_card(recent)
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": body.text}
                ]
                tracer.attach("system_prompt_preview", system_content[:800])
                tracer.attach("messages_preview", messages)
            
            model = FT_MODEL or BASE_MODEL
            
            debugbus.emit(turn_id, "prompt", "prompt_ready", {
                "system_len": len(system_content),
                "messages_roles": [m["role"] for m in messages],
                "est_tokens": estimate_tokens(messages)
            })
            
            # Emit start event
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
            
            # Stream from OpenAI
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = apply_live_settings({
                "model": model,
                "messages": messages,
                "stream": True
            })
            
            full_reply = ""
            first_token_ms = None
            t0 = time.perf_counter()
            
            with tracer.span("openai:stream", {"model": model}):
                async with _http_client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as r:
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data: "):
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
                        except Exception as parse_err:
                            # Don't crash on parse errors, just log
                            print(f"[STREAM] Parse error: {parse_err}")
                            yield f'data: {json.dumps({"type":"debug","raw":line[:200]})}\n\n'
            
            total_ms = int((time.perf_counter() - t0) * 1000)
            
            # Persist to Supabase
            if full_reply:
                try:
                    await conv.save_turn(user_id, thread_id, "user", body.text)
                    await conv.save_turn(user_id, thread_id, "assistant", full_reply)
                except Exception as e:
                    print(f"[STORE] Failed to save streaming turn: {e}")
                    # Don't fail the stream if save fails
            
            # Update in-memory history and generate momentum
            hist.append({"role": "assistant", "content": full_reply})
            
            try:
                mom = generate_momentum(
                    history=hist,
                    current_user_text=body.text,
                    max_turns=6,
                    temperature=0.8,
                    max_tokens=220,
                    api_key=OPENAI_API_KEY
                )
                momentum_block = {
                    "ideas": mom.get("ideas", []),
                    "debug": mom.get("debug", {}),
                }
            except Exception as e:
                print(f"[MOMENTUM] Generation failed: {e}")
                momentum_block = {"ideas": [], "debug": {"error": str(e)[:200]}}
            
            # Emit completion
            debugbus.emit(turn_id, "complete", "turn_end", {
                "total_ms": total_ms,
                "chars": len(full_reply)
            })
            
            tracer.attach("final_text_preview", full_reply[:800])
            meta = {"timing_ms": {"first_token": first_token_ms, "total": total_ms}}
            
            full_debug = tracer.to_payload()
            full_debug["momentum"] = momentum_block
            
            yield f'data: {json.dumps({"type":"complete","turn_id":turn_id,"final_text":full_reply,"meta":meta,"messages":messages,"prompt":build_prompt_payload(messages),"debug":full_debug,"debug_events":debugbus.dump(turn_id)})}\n\n'
            
            debugbus.clear(turn_id)
            tracer.finalize(status="ok", extra=meta)
        
        except Exception as e:
            err = str(e)[:300]
            print(f"[STREAM] Error: {err}")
            tracer.event("pipeline:error", {"error": err}, level="error")
            debugbus.emit(turn_id, "error", "pipeline_error", {"error": err})
            yield f'data: {json.dumps({"type":"error","error":err,"debug":tracer.to_payload(),"debug_events":debugbus.dump(turn_id)})}\n\n'
            debugbus.clear(turn_id)
            tracer.finalize(status="error", error=err)
    
    return StreamingResponse(event_gen(), media_type="text/event-stream")

# ---------- Settings Endpoints ----------
@app.get("/settings")
async def get_settings():
    """Get current live settings"""
    return LIVE.model_dump()

@app.post("/settings")
async def update_settings(body: SettingsIn):
    """Update live settings without restart"""
    global LIVE
    
    data = LIVE.model_dump()
    for k, v in body.model_dump(exclude_none=True).items():
        data[k] = v
    
    try:
        updated = LiveSettings(**data)
    except ValidationError as e:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_settings", "details": e.errors()}
        )
    
    LIVE = updated
    return LIVE.model_dump()

@app.get("/metrics")
async def metrics():
    """Health check endpoint"""
    return {"ok": True, "model": FT_MODEL or BASE_MODEL}

# ---------- Startup ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
    "server:app",
    host="0.0.0.0",
    port=PORT,
    http="httptools"
)