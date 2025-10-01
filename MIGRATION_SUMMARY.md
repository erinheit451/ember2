# Migration Summary

## What Changed

### Old Structure (Root Level)
```
ember/
├── server.py
├── conversation_store.py
├── prompt_card.py
├── debug.py
├── momentum.py
├── supabase_client.py
├── index.html
├── .env
└── requirements.txt
```

### New Structure (Organized)
```
ember/
├── backend/
│   ├── app/
│   │   ├── main.py              (was: server.py)
│   │   ├── config.py            (new: centralized config)
│   │   ├── database/
│   │   │   └── supabase_client.py
│   │   ├── services/
│   │   │   ├── conversation.py  (was: conversation_store.py)
│   │   │   ├── prompt_builder.py (was: prompt_card.py)
│   │   │   └── momentum.py
│   │   └── utils/
│   │       └── debug.py
│   ├── .env
│   └── requirements.txt
└── frontend/
    └── index.html
```

## Import Changes

### Before
```python
import conversation_store as conv
import prompt_card as card
from debug import DebugTracer
from momentum import generate_momentum
from supabase_client import supabase
```

### After
```python
from app.services import conversation
from app.services import prompt_builder
from app.services.momentum import generate_momentum
from app.utils.debug import DebugTracer
from app.database.supabase_client import supabase
from app.config import OPENAI_API_KEY, FT_MODEL, BASE_MODEL
```

## Key Improvements

1. **Separation of Concerns**
   - Database logic in `database/`
   - Business logic in `services/`
   - Utilities in `utils/`
   - Configuration in `config.py`

2. **Environment Management**
   - All env vars loaded in one place (`config.py`)
   - No more scattered `load_dotenv()` calls

3. **Frontend Separation**
   - Frontend code lives in `frontend/`
   - Backend code lives in `backend/`
   - Clear boundary between client and server

4. **Better Imports**
   - Explicit module paths
   - Clear dependencies
   - Easier to navigate

5. **Professional Structure**
   - Industry-standard layout
   - Easy to scale
   - Clear for new developers

## How to Run

```bash
cd backend
python -m app.main
```

Visit: http://localhost:8000

## Files You Can Delete (After Verification)

Once you've verified the new structure works, you can delete:
- `server.py`
- `conversation_store.py`
- `prompt_card.py`
- `debug.py`
- `momentum.py`
- `supabase_client.py`
- `index.html` (root level)
- `.env` (root level)
- `requirements.txt` (root level)
- `test_supabase.py` (if you want - was just for testing)

Keep:
- `ember.db` (if you need old SQLite data)
- Archive folders
- Simlab (separate project)
- Git files
