"""
Debug tracing utilities for Ember.
"""
import os, json, time, uuid, threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

# --------- safe serialization & redaction ---------
_REDACT_KEYS = {"authorization", "api_key", "apikey", "password", "secret", "bearer"}

def _truncate(val, max_len=2000):
    if isinstance(val, str) and len(val) > max_len:
        return val[:max_len] + f"…({len(val)-max_len} more)"
    if isinstance(val, list) and len(val) > 50:
        return val[:50] + [f"...(+{len(val)-50} more)"]
    return val

def _safe(obj: Any) -> Any:
    try:
        if obj is None: return None
        if isinstance(obj, (bool, int, float, str)): return _truncate(obj)
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                k_str = str(k)
                if k_str.lower() in _REDACT_KEYS:
                    out[k_str] = "[REDACTED]"
                else:
                    out[k_str] = _safe(v)
            return out
        if isinstance(obj, (list, tuple)):
            return [_safe(x) for x in _truncate(list(obj))]
        # fallback
        return _truncate(str(obj))
    except Exception as _:
        return "[UNSERIALIZABLE]"

# --------- tracer ---------
class DebugTracer:
    """
    Records ordered steps/spans/events for one turn.
    - Console: prints JSON lines (easy to grep)
    - File: writes to data/traces/YYYY-MM-DD.jsonl (append-only)
    - UI: you can include tracer.to_payload() in SSE 'start'/'complete' events
    """
    def __init__(self, user_id: str, thread_id: str):
        self.trace_id = str(uuid.uuid4())
        self.user_id = user_id
        self.thread_id = thread_id
        self.turn_id = f"turn_{int(time.time()*1000)}_{self.trace_id[:8]}"
        self.t0 = time.perf_counter()
        self.steps: List[Dict[str, Any]] = []
        self.attachments: Dict[str, Any] = {}
        self._lock = threading.Lock()
        os.makedirs("data/traces", exist_ok=True)

        self._console({"evt": "trace_start", "trace_id": self.trace_id,
                       "user_id": user_id, "thread_id": thread_id, "turn_id": self.turn_id})

    def _now_ms(self): return int((time.perf_counter() - self.t0) * 1000)

    def event(self, name: str, data: Optional[Dict[str, Any]] = None, level: str = "info"):
        with self._lock:
            step = {
                "name": name,
                "t_ms": self._now_ms(),
                "level": level,
                "data": _safe(data) if data else {}
            }
            self.steps.append(step)
            self._console({"evt": "step", "turn_id": self.turn_id, "name": name, "level": level})

    @contextmanager
    def span(self, name: str, data: Optional[Dict[str, Any]] = None, level: str = "info"):
        start = self._now_ms()
        if data: self.event(f"{name}:start", data, level=level)
        else:    self.event(f"{name}:start", level=level)
        try:
            yield
            dur = self._now_ms() - start
            self.event(f"{name}:end", {"dur_ms": dur}, level=level)
        except Exception as e:
            dur = self._now_ms() - start
            self.event(f"{name}:error", {"dur_ms": dur, "err": str(e)[:300]}, level="error")
            raise

    def attach(self, key: str, value: Any):
        with self._lock:
            self.attachments[key] = _safe(value)

    def to_payload(self) -> Dict[str, Any]:
        # What you send to the UI
        return {
            "trace_id": self.trace_id,
            "turn_id": self.turn_id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "steps": self.steps,
            "attachments": self.attachments
        }

    def finalize(self, status: str = "ok", error: Optional[str] = None, extra: Optional[Dict[str,Any]] = None):
        total_ms = self._now_ms()
        payload = {
            "ts": int(time.time()),
            "trace_id": self.trace_id,
            "turn_id": self.turn_id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "status": status,
            "error": error[:300] if error else None,
            "total_ms": total_ms,
            "steps": self.steps,
            "attachments": self.attachments,
            "extra": _safe(extra) if extra else None
        }
        self._console({"evt": "trace_end", "turn_id": self.turn_id, "status": status, "total_ms": total_ms})
        self._file(payload)

    # -------- sinks --------
    def _console(self, obj: Dict[str, Any]):
        print(json.dumps(obj, ensure_ascii=False))

    def _file(self, obj: Dict[str, Any]):
        fname = time.strftime("data/traces/%Y-%m-%d.jsonl")
        with open(fname, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
