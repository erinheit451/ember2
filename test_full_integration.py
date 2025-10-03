"""
Test full integration - simulate real chat request through the API
"""
import asyncio
import json
import sys
sys.path.insert(0, 'backend')

from app.main import app
from fastapi.testclient import TestClient

def test_streaming_with_goals():
    print("=" * 70)
    print("FULL INTEGRATION TEST: Goal System in Streaming Endpoint")
    print("=" * 70)

    client = TestClient(app)

    # Simulate a user message
    import uuid
    test_message = {
        "text": "I'm thinking about getting a dog",
        "user_id": str(uuid.uuid4()),
        "thread_id": str(uuid.uuid4())
    }

    print(f"\nSending request to /chat/stream")
    print(f"Message: {test_message['text']}")
    print(f"User ID: {test_message['user_id']}")
    print(f"Thread ID: {test_message['thread_id']}")

    # Make streaming request
    with client.stream("POST", "/chat/stream", json=test_message) as response:
        print(f"\nResponse status: {response.status_code}")

        if response.status_code != 200:
            print(f"ERROR: Status {response.status_code}")
            print(f"Response: {response.text}")
            return False

        print("\nStreaming events received:")
        print("-" * 70)

        events_found = {
            "start": False,
            "goal_formed": False,
            "goal_progress": False,
            "goal_prompt": False,
            "content": False,
            "complete": False
        }

        for line in response.iter_lines():
            if not line:
                continue

            # SSE format: "data: {json}"
            if line.startswith("data: "):
                data = line[6:]
                try:
                    evt = json.loads(data)
                    evt_type = evt.get("type")

                    if evt_type == "start":
                        events_found["start"] = True
                        print(f"\n[START] Model: {evt.get('model', 'unknown')}")

                    elif evt_type == "debug_step":
                        step_name = evt.get("step", "")
                        print(f"\n[DEBUG_STEP] {step_name}")

                        if "Goal Formed" in step_name:
                            events_found["goal_formed"] = True
                            goal_data = evt.get("data", {})
                            print(f"  Goal: {goal_data.get('goal_text', 'N/A')[:80]}...")
                            print(f"  Strategy: {goal_data.get('strategy', 'N/A')}")

                        elif "Goal Progress" in step_name:
                            events_found["goal_progress"] = True
                            progress_data = evt.get("data", {})
                            print(f"  Engagement: {progress_data.get('engagement_type', 'N/A')}")
                            print(f"  Progress: {progress_data.get('progress', 'N/A')}%")
                            print(f"  Next Strategy: {progress_data.get('next_strategy', 'N/A')}")

                        elif "Goal-Driven Prompt" in step_name:
                            events_found["goal_prompt"] = True
                            prompt_data = evt.get("data", {})
                            print(f"  Strategy: {prompt_data.get('strategy', 'N/A')}")

                    elif evt_type == "content":
                        if not events_found["content"]:
                            print(f"\n[CONTENT] Streaming response...")
                            events_found["content"] = True

                    elif evt_type == "complete":
                        events_found["complete"] = True
                        print(f"\n[COMPLETE] Final text length: {len(evt.get('final_text', ''))} chars")

                except json.JSONDecodeError:
                    continue

        print("\n" + "=" * 70)
        print("EVENT CHECK:")
        print("=" * 70)

        all_passed = True
        for event, found in events_found.items():
            status = "PASS" if found else "FAIL"
            print(f"{status:4} - {event}")
            if not found and event in ["goal_formed", "goal_progress", "goal_prompt"]:
                all_passed = False

        print("=" * 70)

        if all_passed:
            print("\nSUCCESS: All goal system events detected!")
            return True
        else:
            print("\nFAILURE: Some goal events missing!")
            print("This means goals are not showing in the debug panel.")
            return False

if __name__ == "__main__":
    try:
        success = test_streaming_with_goals()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
