"""
Test the goal tracking system
"""
import asyncio
import sys
sys.path.insert(0, 'backend')

from app.services.goal_formation import form_conversation_goal
from app.services.goal_progress import assess_goal_progress
from app.services.goal_tracker import save_goal, get_goal, update_goal_progress

async def test_goal_system():
    print("[TEST] Goal Tracking System\n")

    # Test 1: Goal Formation
    print("=" * 60)
    print("TEST 1: Goal Formation")
    print("=" * 60)

    history = [
        {"role": "user", "content": "I'm trying to decide between two job offers but I can't make up my mind."},
    ]
    current_message = "One pays more but the other seems more interesting. I don't know what to do."
    thread_id = "test-thread-123"

    print("\nConversation context:")
    print(f"History: {history[0]['content']}")
    print(f"Current: {current_message}")

    print("\nForming goal...")
    goal = await form_conversation_goal(history, current_message, thread_id)

    print(f"\nGoal formed:")
    print(f"   Goal: {goal.goal_text}")
    print(f"   Real issue: {goal.real_issue}")
    print(f"   Insight needed: {goal.insight_needed}")
    print(f"   Why avoiding: {goal.why_avoiding}")
    print(f"   Strategy: {goal.strategy}")

    # Test 2: Save Goal
    print("\n" + "=" * 60)
    print("TEST 2: Save Goal to Database")
    print("=" * 60)

    try:
        print("\nSaving goal...")
        result = await save_goal(goal)
        if result:
            print(f"SUCCESS: Goal saved! ID: {result.get('id', 'N/A')}")
        else:
            print("WARNING: Goal save returned empty (table might not exist)")
    except Exception as e:
        print(f"ERROR saving goal: {e}")
        print("   -> You need to create the conversation_goals table in Supabase")
        print("   -> Run the SQL from backend/schema/conversation_goals.sql")

    # Test 3: Goal Progress Assessment
    print("\n" + "=" * 60)
    print("TEST 3: Assess Goal Progress")
    print("=" * 60)

    user_response = "Yeah, I mean... I guess I'm worried about making the wrong choice and regretting it."

    print(f"\nUser's response: {user_response}")
    print("\nAssessing progress...")

    progress = await assess_goal_progress(goal, user_response)

    print(f"\nProgress assessment:")
    print(f"   Engagement: {progress.get('engagement_type')}")
    print(f"   Progress: {progress.get('progress')}%")
    print(f"   Reasoning: {progress.get('progress_reasoning')}")
    print(f"   Next strategy: {progress.get('next_strategy')}")

    # Test 4: Retrieve Goal
    print("\n" + "=" * 60)
    print("TEST 4: Retrieve Goal from Database")
    print("=" * 60)

    try:
        print(f"\nRetrieving goal for thread: {thread_id}")
        retrieved_goal = await get_goal(thread_id)
        if retrieved_goal:
            print(f"SUCCESS: Goal retrieved!")
            print(f"   Goal: {retrieved_goal.goal_text}")
            print(f"   Progress: {retrieved_goal.progress}%")
        else:
            print("WARNING: No goal found (table might not exist or goal wasn't saved)")
    except Exception as e:
        print(f"ERROR retrieving goal: {e}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    print("\nIf you saw database errors, you need to:")
    print("1. Go to Supabase SQL Editor")
    print("2. Run the SQL from backend/schema/conversation_goals.sql")
    print("3. Re-run this test")

if __name__ == "__main__":
    asyncio.run(test_goal_system())
