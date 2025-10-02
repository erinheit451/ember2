"""
Conversation goal tracking and management
"""
from typing import Dict, Optional, List
from datetime import datetime
from app.database.supabase_client import supabase

class ConversationGoal:
    def __init__(
        self,
        thread_id: str,
        goal_text: str,
        real_issue: str,
        insight_needed: str,
        why_avoiding: str,
        strategy: str = "initial",
        progress: int = 0,
        turns_pursuing: int = 0,
        created_at: str = None,
        last_updated: str = None
    ):
        self.thread_id = thread_id
        self.goal_text = goal_text  # "Help them see X"
        self.real_issue = real_issue  # "They're avoiding Y"
        self.insight_needed = insight_needed  # "They need to understand Z"
        self.why_avoiding = why_avoiding  # "Because W"
        self.strategy = strategy  # Current approach
        self.progress = progress  # 0-100
        self.turns_pursuing = turns_pursuing
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.last_updated = last_updated or datetime.utcnow().isoformat()

    def to_dict(self):
        return {
            "thread_id": self.thread_id,
            "goal_text": self.goal_text,
            "real_issue": self.real_issue,
            "insight_needed": self.insight_needed,
            "why_avoiding": self.why_avoiding,
            "strategy": self.strategy,
            "progress": self.progress,
            "turns_pursuing": self.turns_pursuing,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**{k: v for k, v in data.items() if k != 'id'})

async def save_goal(goal: ConversationGoal) -> Dict:
    """Save or update conversation goal in Supabase"""
    try:
        # Check if goal exists
        existing = supabase.table("conversation_goals") \
            .select("*") \
            .eq("thread_id", goal.thread_id) \
            .execute()

        goal_data = goal.to_dict()

        if existing.data:
            # Update existing
            result = supabase.table("conversation_goals") \
                .update(goal_data) \
                .eq("thread_id", goal.thread_id) \
                .execute()
        else:
            # Insert new
            result = supabase.table("conversation_goals") \
                .insert(goal_data) \
                .execute()

        return result.data[0] if result.data else {}
    except Exception as e:
        print(f"[GOAL_TRACKER] Error saving goal: {e}")
        return {}

async def get_goal(thread_id: str) -> Optional[ConversationGoal]:
    """Retrieve conversation goal from Supabase"""
    try:
        result = supabase.table("conversation_goals") \
            .select("*") \
            .eq("thread_id", thread_id) \
            .execute()

        if result.data:
            return ConversationGoal.from_dict(result.data[0])
        return None
    except Exception as e:
        print(f"[GOAL_TRACKER] Error getting goal: {e}")
        return None

async def update_goal_progress(
    thread_id: str,
    progress: int,
    strategy: str,
    turns_pursuing: int
) -> Dict:
    """Update goal progress metrics"""
    try:
        result = supabase.table("conversation_goals") \
            .update({
                "progress": progress,
                "strategy": strategy,
                "turns_pursuing": turns_pursuing,
                "last_updated": datetime.utcnow().isoformat()
            }) \
            .eq("thread_id", thread_id) \
            .execute()

        return result.data[0] if result.data else {}
    except Exception as e:
        print(f"[GOAL_TRACKER] Error updating progress: {e}")
        return {}
