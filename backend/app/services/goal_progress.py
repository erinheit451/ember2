"""
Assess progress toward conversation goals
"""
import json
from typing import Dict
from openai import OpenAI
from app.config import OPENAI_API_KEY, BASE_MODEL
from app.services.goal_tracker import ConversationGoal

PROGRESS_ASSESSMENT_PROMPT = """You are assessing progress toward a conversation goal.

GOAL: {goal_text}
REAL ISSUE: {real_issue}
INSIGHT NEEDED: {insight_needed}
CURRENT STRATEGY: {strategy}
TURNS PURSUING: {turns_pursuing}

User's latest response: {user_message}

Assess:
1. Did they engage with the goal? (resist, deflect, go deeper, breakthrough)
2. How much progress? (0-100)
3. What strategy should we use next?

Return ONLY JSON:
{{
  "engagement_type": "<resist|deflect|engage|breakthrough>",
  "progress": <0-100>,
  "progress_reasoning": "<one sentence explaining the score>",
  "next_strategy": "<specific strategy to use next turn>",
  "goal_achieved": <true|false>
}}

Strategy options:
- "validate_then_redirect" - acknowledge their point, then push toward goal
- "direct_naming" - directly name what they're avoiding
- "personal_sharing" - share related experience to create safety
- "external_perspective" - bring in framework/domain to shift view
- "pattern_highlighting" - point out what they keep doing
- "pivot_approach" - try completely different angle
- "deepen_current" - they're engaging, go deeper on same thread
- "goal_achieved" - they got it, evolve to new goal
"""

async def assess_goal_progress(
    goal: ConversationGoal,
    user_message: str
) -> Dict:
    """
    Assess how user's response relates to goal progress
    """
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = PROGRESS_ASSESSMENT_PROMPT.format(
            goal_text=goal.goal_text,
            real_issue=goal.real_issue,
            insight_needed=goal.insight_needed,
            strategy=goal.strategy,
            turns_pursuing=goal.turns_pursuing,
            user_message=user_message
        )

        response = client.chat.completions.create(
            model=BASE_MODEL,
            messages=[
                {"role": "system", "content": "You assess conversation goal progress. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown
        if raw.startswith("```json"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        elif raw.startswith("```"):
            raw = raw.replace("```", "").strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[GOAL_PROGRESS] JSON parse error: {e}")
            print(f"[GOAL_PROGRESS] Raw response: {raw}")
            return {
                "engagement_type": "unknown",
                "progress": goal.progress,  # Keep current
                "progress_reasoning": "Failed to parse progress",
                "next_strategy": goal.strategy,  # Keep current
                "goal_achieved": False,
                "error": str(e)
            }
    except Exception as e:
        print(f"[GOAL_PROGRESS] Error assessing progress: {e}")
        return {
            "engagement_type": "error",
            "progress": goal.progress,  # Keep current
            "progress_reasoning": "Error assessing progress",
            "next_strategy": goal.strategy,  # Keep current
            "goal_achieved": False,
            "error": str(e)
        }
