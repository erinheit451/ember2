"""
Conversation storage service using Supabase.
"""
from typing import List, Dict, Optional
from datetime import datetime
from app.database.supabase_client import supabase

async def save_turn(
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict] = None
) -> Dict:
    """
    Save a message turn to Supabase
    """
    # Check if conversation exists - use execute() instead of maybe_single()
    conv = supabase.table("conversations") \
        .select("id") \
        .eq("id", conversation_id) \
        .execute()

    if not conv.data:  # No conversation found
        # Create conversation
        supabase.table("conversations").insert({
            "id": conversation_id,
            "user_id": user_id,
            "title": content[:50] if role == "user" else "New conversation",
            "metadata": {}
        }).execute()
    else:
        # Update conversation timestamp
        supabase.table("conversations").update({
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", conversation_id).execute()

    # Insert message
    message_data = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "metadata": metadata or {}
    }

    if embedding:
        message_data["embedding"] = embedding

    result = supabase.table("messages").insert(message_data).execute()

    if not result.data:
        raise Exception("Failed to insert message")

    return result.data[0]

async def get_recent(
    user_id: str,
    conversation_id: str,
    limit: int = 20
) -> List[Dict]:
    """
    Get recent messages from a conversation
    """
    result = supabase.table("messages") \
        .select("role, content, created_at, metadata") \
        .eq("conversation_id", conversation_id) \
        .eq("user_id", user_id) \
        .order("created_at", desc=False) \
        .limit(limit) \
        .execute()

    return result.data

async def get_all_conversations(user_id: str) -> List[Dict]:
    """
    Get all conversations for a user
    """
    result = supabase.table("conversations") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("updated_at", desc=True) \
        .execute()

    return result.data

async def search_messages(
    user_id: str,
    query_embedding: List[float],
    threshold: float = 0.7,
    limit: int = 5
) -> List[Dict]:
    """
    Semantic search across all user messages
    """
    result = supabase.rpc("match_messages", {
        "query_embedding": query_embedding,
        "match_threshold": threshold,
        "match_count": limit
    }).execute()

    return result.data

async def log_usage(
    user_id: str,
    conversation_id: str,
    message_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    metadata: Optional[Dict] = None
) -> Dict:
    """
    Log token usage for billing
    """
    total_tokens = prompt_tokens + completion_tokens

    # Cost calculation (adjust based on your model pricing)
    # Example: gpt-4o-mini pricing
    prompt_cost = (prompt_tokens / 1000) * 0.15  # $0.15 per 1k tokens
    completion_cost = (completion_tokens / 1000) * 0.60  # $0.60 per 1k tokens
    total_cost_cents = int((prompt_cost + completion_cost) * 100)

    result = supabase.table("usage_logs").insert({
        "user_id": user_id,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_cents": total_cost_cents,
        "model": model,
        "metadata": metadata or {}
    }).execute()

    return result.data[0] if result.data else {}

def init_db():
    """No-op for Supabase (tables already exist)"""
    print("[SUPABASE] Connected to database")
