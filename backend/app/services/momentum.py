"""
Momentum generation service for Ember.
"""
import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import OPENAI_API_KEY, MOMENTUM_MODEL

def _client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key, timeout=30.0, max_retries=1)

def _format_history(history: List[Dict[str, str]], max_turns: int) -> str:
    """
    Keep the last N messages; no aggressive trimming. Preserve order.
    Expected schema per item: {"role": "user"|"assistant", "content": "text"}
    """
    use = history[-max_turns:] if max_turns and len(history) > max_turns else history[:]
    lines = []
    for m in use:
        role = "User" if m.get("role") == "user" else "Assistant"
        text = (m.get("content") or "").strip()
        lines.append(f"{role}: {text}")
    return "\n".join(lines)

SPARK_PROMPT = """You generate 2–3 SPARKS that move a chat forward with honesty and nerve.
They are NOT full replies. Each spark is a short opening move (≤16 words) that shifts perspective,
pursues subtext, or makes a precise, vivid ask. No platitudes. No generic advice verbs.

Context (recent exchange):
{history_block}

Quality bar (contrastive examples):
User: "I might hit a strip club. Been coding all day."
Good:  "Wild can help, but what are you actually hungry for tonight?"
Better:"Screen-stuck to neon. Is it thrill you want—or to be seen?"
Bad:   "That's a big jump. Have you considered calling a friend?"

User: "He's been vomiting."
Good:  "Okay. What changed in the last 24 hours—food, stress, routine?"
Better:"What's the *one* weird detail you almost skipped? That's often the clue."
Bad:   "Vomiting can be serious. Consider calling a vet."

Rules:
- Do NOT mirror the user's phrasing.
- Each spark must feel specific to this conversation's subtext.
- Allowed angles: "reframe", "micro-provocation", "specificity", "story-pivot".
- Keep each spark's seed ≤16 words.

Return ONLY JSON:
{{"ideas":[
  {{"seed":"...", "angle":"reframe|micro-provocation|specificity|story-pivot", "why":"one short sentence"}},
  {{"seed":"...", "angle":"...", "why":"..."}}
]}}
"""

def momentum_sparks(history_block: str, api_key: Optional[str] = None) -> dict:
    client = _client(api_key)
    r = client.chat.completions.create(
        model=MOMENTUM_MODEL,
        messages=[
            {"role":"system","content":"You produce lean, subtext-savvy conversation sparks."},
            {"role":"user","content": SPARK_PROMPT.format(history_block=history_block)}
        ],
        temperature=0.8, max_tokens=220
    )
    raw = r.choices[0].message.content.strip()

    # Strip markdown code blocks if present
    json_text = raw
    if raw.startswith("```json"):
        json_text = raw.replace("```json", "").replace("```", "").strip()
    elif raw.startswith("```"):
        json_text = raw.replace("```", "").strip()

    # safe parse
    try:
        data = json.loads(json_text)
        if "ideas" not in data:
            data = {"ideas": [], "raw": raw, "error": "Missing 'ideas' key in response"}
    except Exception as e:
        data = {"ideas": [], "raw": raw, "error": f"JSON parse error: {str(e)}"}
    data["debug"] = {"model": MOMENTUM_MODEL, "raw_content": raw}
    return data

def generate_momentum(
    history: List[Dict[str, str]],
    current_user_text: str,
    *,
    model: Optional[str] = None,
    max_turns: int = 6,
    temperature: float = 0.8,
    max_tokens: int = 220,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate 2–3 momentum ideas using the new SPARK format.
    Returns: {"ideas":[{...}], "debug": {...}}
    """
    t0 = time.perf_counter()

    # Format history for the new prompt
    history_text = _format_history(history, max_turns=max_turns)

    try:
        # Use the new spark-based momentum generation
        result = momentum_sparks(history_text, api_key=api_key)
        ideas = result.get("ideas", [])

        # Transform the new format to match the expected output format
        transformed_ideas = []
        for idea in ideas:
            transformed_idea = {
                "draft": idea.get("seed", ""),
                "why": idea.get("why", ""),
                "angle": idea.get("angle", "")
            }
            transformed_ideas.append(transformed_idea)

        ideas = transformed_ideas
        parse_error = None

    except Exception as e:
        ideas = []
        parse_error = f"{type(e).__name__}: {e}"

    dt_ms = int((time.perf_counter() - t0) * 1000)
    debug = {
        "model": model or MOMENTUM_MODEL,
        "ms": dt_ms,
        "input_turns": len(history),
        "max_turns_used": max_turns,
        "chars_history": len(history_text),
        "chars_current": len(current_user_text or ""),
        "raw_output_preview": "",
        "parse_error": parse_error,
    }
    return {"ideas": ideas, "debug": debug}

# Optional: tiny CLI for quick local testing
if __name__ == "__main__":
    sample_history = [
        {"role": "user", "content": "I think my dog might be sick."},
        {"role": "assistant", "content": "What symptoms are you seeing?"},
        {"role": "user", "content": "He vomited tonight and seems low energy."},
        {"role": "assistant", "content": "How long has it been and did he eat anything unusual?"}
    ]
    current = "He's just lying around now. I'm a little worried."
    out = generate_momentum(sample_history, current)
    print(json.dumps(out, indent=2))
