-- Conversation Goals Table
-- Tracks persistent conversation goals across multiple turns

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

-- Index for fast lookups by thread_id
CREATE INDEX IF NOT EXISTS idx_conversation_goals_thread ON conversation_goals(thread_id);

-- Index for tracking progress across conversations
CREATE INDEX IF NOT EXISTS idx_conversation_goals_progress ON conversation_goals(progress);
