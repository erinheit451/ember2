# Verification Checklist

## ✅ Completed Tasks

- [x] Created new directory structure (`backend/app/` and subdirectories)
- [x] Created all `__init__.py` files
- [x] Created `config.py` for centralized settings
- [x] Moved `supabase_client.py` to `backend/app/database/`
- [x] Moved `conversation_store.py` to `backend/app/services/conversation.py`
- [x] Moved `prompt_card.py` to `backend/app/services/prompt_builder.py`
- [x] Moved `momentum.py` to `backend/app/services/`
- [x] Moved `debug.py` to `backend/app/utils/`
- [x] Moved and updated `server.py` to `backend/app/main.py`
- [x] Moved `index.html` to `frontend/`
- [x] Copied `.env` and `requirements.txt` to `backend/`
- [x] Created `.gitignore`
- [x] Updated all imports to use new structure
- [x] Fixed `save_turn()` function in conversation service
- [x] Server imports successfully
- [x] Server starts without errors

## 🧪 Test the Server

1. **Import test:**
   ```bash
   cd backend
   python -c "import app.main; print('Success')"
   ```
   Expected: `Success` (with Supabase connection message)

2. **Start server:**
   ```bash
   cd backend
   python -m app.main
   ```
   Expected: Server starts on http://0.0.0.0:8000

3. **Access UI:**
   - Open browser to http://localhost:8000
   - You should see the index.html page

4. **Test API:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello", "user_id": "test", "thread_id": "test-1"}'
   ```
   Expected: JSON response with reply and debug info

## 📋 Manual Verification Steps

1. **Check file structure:**
   ```bash
   ls -R backend/app
   ```
   Should show: config.py, main.py, database/, services/, utils/

2. **Check imports work:**
   ```bash
   cd backend
   python -c "from app.config import OPENAI_API_KEY; print('Config OK')"
   python -c "from app.services import conversation; print('Services OK')"
   python -c "from app.utils.debug import DebugTracer; print('Utils OK')"
   ```

3. **Check environment:**
   ```bash
   cat backend/.env | head -3
   ```
   Should show your environment variables

4. **Check frontend:**
   ```bash
   ls frontend/
   ```
   Should show: index.html

## 🎯 Next Steps

1. **Test a real conversation:**
   - Start the server
   - Open the UI
   - Send a message
   - Verify it stores in Supabase

2. **Clean up old files (optional):**
   - Delete root-level .py files after confirming everything works
   - Keep archive/ and simlab/ folders

3. **Update any external scripts:**
   - If you have deployment scripts, update paths
   - Update any documentation that references old paths

## ✨ Success Criteria

- ✅ Server starts without errors
- ✅ All imports resolve correctly
- ✅ API endpoints respond
- ✅ Frontend loads
- ✅ Conversations save to Supabase
- ✅ Debug tracing works
- ✅ Momentum generation works

## 🐛 Troubleshooting

### "No module named 'app'"
→ Make sure you're running from `backend/` directory

### "Cannot find .env file"
→ Ensure `backend/.env` exists with all required variables

### "Import error: supabase"
→ Install dependencies: `pip install -r backend/requirements.txt`

### Frontend not loading
→ Check that `frontend/index.html` exists
→ Server should serve it from root path `/`
