"""
Goal formation agent - analyzes conversation to establish independent goals
"""
import json
from typing import Dict, List
from openai import OpenAI
from app.config import OPENAI_API_KEY, BASE_MODEL
from app.services.goal_tracker import ConversationGoal

GOAL_FORMATION_PROMPT = """You are analyzing a conversation to establish an independent goal.

Your job: Identify what this person is REALLY struggling with (not just what they're saying),
and decide what you want them to understand by the end of this conversation.

CRITICAL RULES:
- Don't just "support" them - have an actual agenda
- Your goal should be something they need to SEE, not just feel validated about
- Be specific about what they're avoiding and why
- This is a MULTI-TURN goal, not just responding to current message

Conversation so far:
{conversation_history}

Current message: {current_message}

Return ONLY JSON:
{{
  "real_issue": "<what they're actually struggling with underneath their words>",
  "goal_text": "<what you want them to understand/see by end of conversation>",
  "insight_needed": "<the specific realization that would help them>",
  "why_avoiding": "<why they're not seeing this on their own>",
  "initial_strategy": "<how to pursue this goal - be specific about approach>"
}}

Examples of GOOD goals:
- "Help them see that 'what if I pick wrong' is really 'what if I love and lose again'"
- "Get them to recognize their people-pleasing is driven by fear of abandonment"
- "Surface that their perfectionism is preventing them from starting at all"

Examples of BAD goals (too generic/supportive):
- "Help them feel better about their situation"
- "Support them through this difficult time"
- "Listen to their concerns"
"""

async def form_conversation_goal(
    history: List[Dict],
    current_message: str,
    thread_id: str
) -> ConversationGoal:
    """
    Analyze conversation and form an independent goal
    """
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Format history
        history_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Ember'}: {m['content']}"
            for m in history[-10:]  # Last 10 turns for context
        ])

        prompt = GOAL_FORMATION_PROMPT.format(
            conversation_history=history_text,
            current_message=current_message
        )

        response = client.chat.completions.create(
            model=BASE_MODEL,
            messages=[
                {"role": "system", "content": "You are a goal formation agent. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=500
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown
        if raw.startswith("```json"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        elif raw.startswith("```"):
            raw = raw.replace("```", "").strip()

        try:
            data = json.loads(raw)
            return ConversationGoal(
                thread_id=thread_id,
                goal_text=data.get("goal_text", ""),
                real_issue=data.get("real_issue", ""),
                insight_needed=data.get("insight_needed", ""),
                why_avoiding=data.get("why_avoiding", ""),
                strategy=data.get("initial_strategy", "initial_exploration"),
                progress=0,
                turns_pursuing=0
            )
        except json.JSONDecodeError as e:
            print(f"[GOAL_FORMATION] JSON parse error: {e}")
            print(f"[GOAL_FORMATION] Raw response: {raw}")
            # Fallback goal if parsing fails
            return ConversationGoal(
                thread_id=thread_id,
                goal_text="Understand what they're really asking about",
                real_issue="Unclear from current context",
                insight_needed="To be determined through conversation",
                why_avoiding="Unknown",
                strategy="exploratory",
                progress=0,
                turns_pursuing=0
            )
    except Exception as e:
        print(f"[GOAL_FORMATION] Error forming goal: {e}")
        # Fallback goal on any error
        return ConversationGoal(
            thread_id=thread_id,
            goal_text="Understand what they're really asking about",
            real_issue="Error forming goal",
            insight_needed="To be determined through conversation",
            why_avoiding="Unknown",
            strategy="exploratory",
            progress=0,
            turns_pursuing=0
        )
