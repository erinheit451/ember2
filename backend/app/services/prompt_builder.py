"""
Prompt building service for Ember.
"""
from typing import List, Dict

BASE_SYSTEM = (
    "You are Ember. Answer plainly. Prefer information in the MEMORY section over guessing.\n\n"
    "STYLE GUARDRAILS\n"
    "- Do NOT repeat the user's phrasing back at them.\n"
    "- Show understanding by rephrasing or by asking a fresh, concrete follow-up.\n"
    "- Keep openings short (≤ 10 words), then add substance.\n\n"
    "Anti-mirror examples:\n"
    "User: \"he's been vomiting\"\n"
    "Bad:  \"Vomiting can be serious.\"\n"
    "Good: \"That can be serious. Did he eat anything unusual?\"\n"
    "User: \"I'm lonely\"\n"
    "Bad:  \"Feeling lonely can be hard.\"\n"
    "Good: \"Who do you normally lean on when it feels quiet?\""
)

def build_memory_card(recent_turns: List[Dict[str, str]], max_items: int = 20) -> str:
    """
    Build a minimal MEMORY block that lists the last N user/assistant messages
    as short bullet snippets. This is NOT 'chat history'; it's a compact card.
    """
    if not recent_turns:
        return ""
    lines = ["--- MEMORY ---", "Recent:"]
    for t in recent_turns[-max_items:]:
        role = t.get("role", "user")
        text = (t.get("content", "") or "").strip()
        if text:
            lines.append(f"- {role}: \"{text}\"")
    lines.append("--- END MEMORY ---")
    return "\n".join(lines)

def build_system_with_card(recent_turns: List[Dict[str, str]]) -> str:
    mem = build_memory_card(recent_turns)
    return BASE_SYSTEM + ("\n\n" + mem if mem else "")
