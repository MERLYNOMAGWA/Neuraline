# Neuraline — Personal Cognitive Growth Engine

Neuraline is an AI-powered personal growth assistant that blends emotional reflection, structured thinking, behavioral coaching, and purpose alignment.
This repository demonstrates a production-capable backend (FastAPI) with RAG, multi-agent orchestration (MCP), memory persistence (Chroma), model routing (Gemini / Groq fallback), conversational memory, and a simple frontend interface.

## 🚀 Project Overview

**Problem statement**: People often know what to do but struggle to act consistently. Neuraline bridges the gap between insight and action by combining emotional intelligence (reflection), cognitive structuring (planning), behavior design (coaching), and long-term purpose alignment.

### Core features

- RAG pipeline with Chroma (document ingestion → embeddings → semantic retrieval)
- Model abstraction & intelligent routing (Gemini preferred, Groq fallback)
- MCP engine coordinating multiple agents: Reflector, Strategist, Coach, Purpose
- Conversation memory and persistence (Chroma-backed or in-memory fallback)
- FastAPI backend exposing endpoints for chat and MCP runs
- Simple frontend (Streamlit/HTML) to demo chat
- Test suite for RAG, conversation, and multi-agent coordination

## 📁 Repo Layout (high-level)
```
neuraline/
├── app/
│   ├── main.py                 # FastAPI app wiring
│   ├── api/
│   │   ├── chat.py             # chat endpoints
│   │   └── mcp.py              # mcp endpoints
│   ├── agents/                 # agent implementations & orchestrator
│   ├── services/
│   │   ├── ai_clients.py       # model clients (Gemini/Groq/embeddings)
│   │   ├── retriever.py        # RAG retriever wrapper
│   │   ├── vector_store.py
│   │   ├── memory/             # e.g. chroma_memory.py
│   │   └── conversation_manager.py
│   └── prompts/                # prompt templates and prompt engineering
├── data/
│   └── sources/                # text sources for RAG (txt files)
├── tests/
│   └── ...                     # unit/manual tests (conversation_test_manual.py)
├── neuraline_frontend.py       # Streamlit UI (optional)
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🏗 Architecture & Data Flow

*(Include a diagram image here docs/architecture.png — use draw.io / mermaid / Lucidchart)*

### High-level flow

1. User sends a text message to FastAPI `/api/v1/chat`.
2. `ConversationManager` classifies the request and loads session memory.
3. If relevant, RAG retriever fetches context from Chroma.
4. `ModelRouter` selects an LLM (Gemini / Groq) and injects context for RAG tasks.
5. For complex tasks, MCP orchestrator triggers multiple agents (Reflector → Strategist → Coach → Purpose) in chain or parallel.
6. Responses are fused into a single Neuraline reply, validated, and saved to memory.
7. Frontend displays reply; persistence is stored in Chroma (or Redis/Postgres in production).

## 🔧 Installation & Local Setup

### Prerequisites

- Python 3.11+
- pip
- (optional) Docker & docker-compose for containerized setup

### Quick start (virtualenv)
```bash
git clone <repo-url>
cd neuraline/backend
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env           # edit keys in .env
```

### Run locally (development)
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# frontend (if using Streamlit)
streamlit run neuraline_frontend.py
```

### Run with Docker Compose
```bash
docker-compose up --build
# backend -> http://localhost:8000/docs
# frontend -> http://localhost:8501
```

## 🔐 Environment Variables

Create a `.env` file (do NOT commit secrets). Example `.env.example`:
```env
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
CHROMA_PERSIST_DIR=./data/chroma
OPENAI_API_KEY=... # if used
FRONTEND_API_URL=http://localhost:8000/api/v1
LOG_LEVEL=INFO
```

<img src="assets/Screenshot 2025-10-25 202103.png" alt="Screenshot of the app interface" width="900"/>

## 📚 API Endpoints (examples)

FastAPI auto-docs (Swagger): `GET /docs` when server runs.

### POST /api/v1/chat/chat

**Request:**
```json
{
  "message": "Im feeling overwhelmed with a lot to do",
  "session_id": "test_user_1"
}
```


**Response:**
<img src="assets/Screenshot 2025-10-25 202732.png" alt="Screenshot of the app interface" width="900"/>

<img src="assets/Screenshot 2025-10-25 202814.png" alt="Screenshot of the app interface" width="900"/>

### POST /api/v1/mcp/run

**Request:**
```json
{
  "query": "How do I stop procrastinating?",
  "session_id": "user_123",
  "mode": "chain",
  "roles": ["reflector","strategist","coach","purpose"]
}
```
<img src="assets/Screenshot 2025-10-25 203836.png" alt="Screenshot of the app interface" width="900"/>

**Response:** 
<img src="assets/Screenshot 2025-10-25 203945.png" alt="Screenshot of the app interface" width="900"/>

## 🧠 RAG Pipeline (how it works)

1. **Ingestion**: `app/tests/ingest_sources.py` reads `data/sources/*.txt`, chunks (LangChain TextSplitter or custom), and stores chunks in Chroma with embeddings.
2. **Embeddings**: Use a local HuggingFace embedder (e.g., `sentence-transformers/all-MiniLM-L6-v2`) or cloud provider.
3. **Storage**: Chroma stores vectors with metadata including source filename and chunk offsets.
4. **Retrieval**: For a query, the retriever computes embedding → nearest-neighbor search → returns top-k contexts with relevance scores.
5. **Prompt integration**: The retrieved text is injected into the final prompt under a `Context:` section before model call.

### Tips:

- Chunk size ~ 500 tokens with 50-100 token overlap.
- Store `source`, `chunk_id`, `start_offset` metadata to later show provenance.
- Persist Chroma to `./data/chroma` in dev; for production use networked storage or hosted DB.

## 🤖 Agent Architecture & MCP

### Agents

- **Reflector** — Emotional intelligence, asks clarifying questions and reframes.
- **Strategist** — Breaks goals into steps, weekly plans, and SMART micro-tasks.
- **Coach** — Behavior & habit design, nudges and accountability suggestions.
- **Purpose** — Maps actions to values and long-term vision.

### MCP Engine

- Supports `chain` and `parallel` modes.
- Each agent gets role-specific prompt + shared context + snapshot of previous agent outputs.
- Results are combined and optionally fused into coherent Neuraline voice using a fusion prompt.
- Retries and local fallbacks included.

Document agent prompts in `app/prompts/templates.py` and store example agent profiles in `app/mcp/mcp_engine.py`.

## ✅ Testing

- Unit/Integration tests live under `app/tests/`.
- Manual helper scripts: `app/tests/conversation_test_manual.py`, `app/tests/multi_agent_test_manual.py`.
- For pytest ensure you run from repo root and PYTHONPATH includes backend:
```bash
cd backend
pytest -q
```

If tests report `ModuleNotFoundError: No module named 'app'`, run from project root and ensure PYTHONPATH or use `python -m` with package path: `python -m app.tests.conversation_test`.

## 🛡 Safety / Privacy / Memory

- **PII filtering**: Sensitive info (emails, phone numbers) is redacted before saving to memory (see `app/services/safety/content_filter.py`).
- **Memory policy**: avoid persisting raw PII. Add validators in `ResponseValidator` before saving.
- **Memory backends**: default persistence is Chroma; swap to Redis/Postgres/LangSmith with the same MemoryStore API.

## 📦 Deployment

- Use Docker + docker-compose for local staging.
- For cloud, preferred: Render / Railway / Cloud Run. Use environment variables for secrets.
- Recommended production upgrades:
  - Use managed vector DB (Chroma Cloud, Pinecone, Weaviate) for scaling
  - Use LangSmith or a hosted memory service for long-term session storage
  - Add monitoring and rate-limiting for API endpoints

## 🔧 Prompt Engineering & Template Library

- Keep prompt templates in `app/prompts/templates.py`.
- Templates include placeholders: `{context}`, `{memory}`, `{user_input}`.
- Use few-shot examples sparingly and prefer system-level instructions for safe behavior.
- Document each template with expected role, token budget, and when to use (reflection vs planning vs rag_query).

## 🧩 n8n & Automation (optional)

- Save workflow JSON exports in `docs/n8n/` for each automation: daily reflection webhook, email summary, nudges.
- Connect n8n webhook to `/api/v1/automation/webhook` and trigger mcp flows for scheduled summaries.

## 📘 Documentation artifacts to include in repo

- `docs/architecture.png` (diagram)
- `docs/agent_architecture.md` (detailed agent responsibilities)
- `docs/rag_pipeline.md` (chunking settings, embedding model, retrieval tuning)
- `docs/deployment.md` (render/railway instructions)
- `docs/n8n/` (workflow JSONs + screenshots)
- `docs/mcp_protocol.md` (MCP message format, request/response examples)

## 🧪 Example usage

### Run server:
```bash
uvicorn app.main:app --reload
```

### Chat (curl):
```bash
curl -X POST "http://localhost:8000/api/v1/chat" -H "Content-Type: application/json" \
-d '{"message":"I feel stuck with my projects","session_id":"demo_user_1"}'
```

### MCP (curl):
```bash
curl -X POST "http://localhost:8000/api/v1/mcp/run" -H "Content-Type: application/json" \
-d '{"query":"How stop procrastinating?","session_id":"demo_user1","mode":"chain"}'
```