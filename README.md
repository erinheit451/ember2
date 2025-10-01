# Ember - Conversational AI with Memory

A FastAPI-based conversational AI system with Supabase backend for persistence.

## Project Structure

```
ember/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── config.py            # Environment configuration
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   └── supabase_client.py  # Supabase connection
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── conversation.py     # Conversation storage
│   │   │   ├── prompt_builder.py   # Prompt construction
│   │   │   └── momentum.py         # Momentum generation
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── debug.py            # Debug tracing
│   ├── requirements.txt
│   └── .env                     # Environment variables
├── frontend/
│   └── index.html              # Web UI
└── .gitignore
```

## Setup

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   Create `backend/.env` with:
   ```env
   OPENAI_API_KEY=your_key_here
   FT_MODEL=your_finetuned_model_id (optional)
   BASE_MODEL=gpt-4o-mini
   MOMENTUM_MODEL=gpt-4o
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_service_key
   SUPABASE_ANON_KEY=your_anon_key
   PORT=8000
   ```

3. **Run the server:**
   ```bash
   cd backend
   python -m app.main
   ```

   Or with uvicorn directly:
   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8000
   ```

## API Endpoints

- `GET /` - Serve the web UI
- `POST /chat` - Non-streaming chat endpoint
- `POST /chat/stream` - Streaming chat endpoint (SSE)
- `GET /settings` - Get current live settings
- `POST /settings` - Update live settings
- `GET /metrics` - Health check

## Development

The project uses a clean modular structure:

- **app/config.py** - All environment variables and configuration
- **app/database/** - Database clients and connections
- **app/services/** - Business logic (conversation, prompts, momentum)
- **app/utils/** - Shared utilities (debug tracing)
- **app/main.py** - FastAPI application and routes

## Key Features

- **Streaming responses** via Server-Sent Events
- **Memory management** with Supabase persistence
- **Momentum generation** for conversation suggestions
- **Debug tracing** with structured logging
- **Live settings** - adjust temperature, tokens, etc. without restart
