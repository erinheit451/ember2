import asyncio
import uuid
from conversation_store import save_turn, get_recent
from supabase_client import supabase

async def test():
    print("Testing Supabase connection...")
    
    # Test connection
    try:
        result = supabase.table("conversations").select("count", count="exact").execute()
        print(f"[OK] Connected to Supabase")
        print(f"   Current conversations: {result.count}")
    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        return
    
    # Generate proper UUIDs
    test_user_id = str(uuid.uuid4())
    test_conv_id = str(uuid.uuid4())
    
    print(f"\nTest IDs:")
    print(f"   User: {test_user_id}")
    print(f"   Conversation: {test_conv_id}")
    
    # Test save
    print("\n[1/3] Testing save_turn...")
    try:
        msg1 = await save_turn(
            user_id=test_user_id,
            conversation_id=test_conv_id,
            role="user",
            content="Hello, this is a test message"
        )
        print(f"[OK] User message saved: {msg1.get('id')}")
        
        msg2 = await save_turn(
            user_id=test_user_id,
            conversation_id=test_conv_id,
            role="assistant",
            content="Hello! This is a test response."
        )
        print(f"[OK] Assistant message saved: {msg2.get('id')}")
    except Exception as e:
        print(f"[FAIL] Save failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test retrieval
    print("\n[2/3] Testing get_recent...")
    try:
        recent = await get_recent(test_user_id, test_conv_id, limit=5)
        print(f"[OK] Query successful: {len(recent)} messages found")
        for i, msg in enumerate(recent, 1):
            print(f"   [{i}] {msg['role']}: {msg['content'][:40]}...")
    except Exception as e:
        print(f"[FAIL] Query failed: {e}")
        return
    
    # Cleanup
    print("\n[3/3] Cleaning up...")
    try:
        supabase.table("messages").delete().eq("conversation_id", test_conv_id).execute()
        supabase.table("conversations").delete().eq("id", test_conv_id).execute()
        print("[OK] Test data cleaned up")
    except Exception as e:
        print(f"[WARN] Cleanup failed: {e}")
    
    print("\n[SUCCESS] All tests passed!")

if __name__ == "__main__":
    asyncio.run(test())