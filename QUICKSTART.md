# Quick Start Guide

## Running Ember

### Method 1: Using python -m (recommended)
```bash
cd backend
python -m app.main
```

### Method 2: Using uvicorn directly
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Testing the Setup

1. **Test imports:**
   ```bash
   cd backend
   python -c "import app.main; print('Success!')"
   ```

2. **Open the UI:**
   - Visit http://localhost:8000 in your browser

3. **Test the API:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello", "user_id": "test", "thread_id": "test-1"}'
   ```

## Common Issues

### Import Errors
- Make sure you're running from the `backend/` directory
- Ensure all `__init__.py` files exist
- Check that `.env` file is in `backend/.env`

### Module Not Found
```bash
# Run from backend/ directory, not project root
cd backend
python -m app.main
```

### Supabase Connection Errors
- Verify your `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in `.env`
- Check that your Supabase tables are set up (conversations, messages, usage_logs)

## Development Workflow

1. Make changes to code in `backend/app/`
2. Server auto-reloads with `--reload` flag
3. Test changes in browser at http://localhost:8000
4. Check logs in console for debug output

## Project Structure Reminders

- **Configuration**: `backend/app/config.py`
- **Main app**: `backend/app/main.py`
- **Services**: `backend/app/services/`
- **Database**: `backend/app/database/`
- **Utils**: `backend/app/utils/`
- **Frontend**: `frontend/index.html`
