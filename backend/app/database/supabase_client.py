"""
Supabase client configuration.
"""
from supabase import create_client, Client
from typing import Optional
from app.config import SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_ANON_KEY

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

# Service role client (bypasses RLS, use for server operations)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def get_user_client(access_token: str) -> Client:
    """
    Create a user-scoped client (respects RLS)
    Use this when you have a user's JWT token
    """
    if not SUPABASE_ANON_KEY:
        raise ValueError("SUPABASE_ANON_KEY must be set for user clients")

    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    client.auth.set_session(access_token, "")  # Set user context
    return client
