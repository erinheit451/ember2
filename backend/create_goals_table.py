"""
Create conversation_goals table in Supabase
Run this script once to set up the database
"""
from app.database.supabase_client import supabase

SQL = """
CREATE TABLE IF NOT EXISTS conversation_goals (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  thread_id TEXT NOT NULL UNIQUE,
  goal_text TEXT NOT NULL,
  real_issue TEXT NOT NULL,
  insight_needed TEXT NOT NULL,
  why_avoiding TEXT NOT NULL,
  strategy TEXT DEFAULT 'initial',
  progress INTEGER DEFAULT 0,
  turns_pursuing INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  last_updated TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversation_goals_thread ON conversation_goals(thread_id);
CREATE INDEX IF NOT EXISTS idx_conversation_goals_progress ON conversation_goals(progress);
"""

try:
    # Execute the SQL
    result = supabase.rpc('exec_sql', {'sql': SQL}).execute()
    print("✅ Table created successfully!")
    print(f"Result: {result}")
except Exception as e:
    print(f"❌ Error creating table: {e}")
    print("\nPlease run this SQL manually in Supabase SQL Editor:")
    print(SQL)
