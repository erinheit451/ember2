import os
from supabase import create_client, Client
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

# Service role client (bypasses RLS, use for server operations)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def get_user_client(access_token: str) -> Client:
    """
    Create a user-scoped client (respects RLS)
    Use this when you have a user's JWT token
    """
    anon_key = os.getenv("SUPABASE_ANON_KEY")
    client = create_client(SUPABASE_URL, anon_key)
    client.auth.set_session(access_token, "")  # Set user context
    return client