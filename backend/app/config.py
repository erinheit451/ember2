"""
Configuration management for Ember backend.
Loads settings from environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from backend directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FT_MODEL = os.getenv("FT_MODEL", "")  # Fine-tuned model ID
BASE_MODEL = os.getenv("BASE_MODEL", "gpt-4o-mini")
MOMENTUM_MODEL = os.getenv("MOMENTUM_MODEL", "gpt-4o")

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# Server Configuration
PORT = int(os.getenv("PORT", "8000"))

# Validation
if not OPENAI_API_KEY:
    print("[CONFIG] WARNING: OPENAI_API_KEY is not set")
if not FT_MODEL:
    print("[CONFIG] WARNING: FT_MODEL is not set (will fallback to BASE_MODEL)")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[CONFIG] WARNING: Supabase credentials not fully configured")
