# Margin: Document Q&A with Retrieval-Augmented Generation

Margin is a chat app for asking questions about your own documents. Upload PDFs, Word files, text or Markdown, ask a question, and get an answer that is **grounded only in those documents**, with inline citations showing exactly which passages it used.

I built the retrieval pipeline from scratch (chunking, embeddings, vector search and prompt construction) rather than using a framework like LangChain or LlamaIndex. I wanted to understand and control every step.

<!-- Add a screenshot or GIF of the app here, for example: ![Margin demo](docs/demo.gif) -->

---

## Features

- **Grounded answers with citations.** Every answer carries `[1] [2]` citation chips, and a Sources panel shows the exact chunks used, with similarity scores.
- **Says "I don't know".** The system prompt forbids answering beyond the retrieved context, so the model reports when the documents don't contain the answer instead of making one up.
- **Multi-format ingestion.** Supports PDF, DOCX, TXT and MD.
- **Local embeddings and vector store.** `sentence-transformers` and ChromaDB run on your machine, so documents are never sent to a third party for indexing.
- **Provider-agnostic LLM layer.** Switch between Claude, OpenAI or a fully local Ollama model with one environment variable.
- **REST API.** A FastAPI backend with auto-generated docs at `/docs`.

---

## How it works

```
INDEXING (on upload)
file ──► extract text ──► overlapping chunks ──► embeddings (384-d) ──► ChromaDB

ANSWERING (per question)
question ──► embedding ──► top-k similar chunks ──► grounded prompt ──► LLM
                                                                          │
                                            answer + cited sources ◄──────┘
```

### Design decisions

| Component | Choice | Why |
|---|---|---|
| Chunking | Character windows with overlap (800 / 150) | Overlap keeps sentences that span a boundary retrievable from either side |
| Embeddings | `all-MiniLM-L6-v2` | Fast, free, runs on CPU; swappable for `all-mpnet-base-v2` when higher recall matters more than speed |
| Vector store | ChromaDB (persistent) | Local, persists to disk, no infrastructure to run |
| Retrieval | Cosine-similarity top-k (k = 4) | A small k keeps prompts focused and cheap; it can be tuned per corpus |
| Generation | Claude API behind an abstraction | Strong instruction-following for grounded answers; the abstraction allows OpenAI or Ollama as drop-in replacements |
| Backend | FastAPI | Async, typed, free interactive API docs |
| Frontend | A single `index.html` | No build step; the backend serves it directly |

---

## Project structure

```
├── backend/
│   ├── config.py         # Central settings, overridable via .env
│   ├── ingest.py         # Load → chunk → embed → store; similarity search
│   ├── rag.py            # Retrieve → build grounded prompt → generate → return sources
│   ├── llm.py            # Provider-agnostic LLM interface (Anthropic / OpenAI / Ollama)
│   ├── main.py           # FastAPI app and endpoints; serves the frontend
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── index.html        # Chat UI, document upload and sources panel
```

### API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/upload` | Ingest a document (extract, chunk, embed, store) |
| `POST` | `/api/chat` | Ask a question; returns the answer and its sources |
| `POST` | `/api/reset` | Clear the vector index |
| `GET` | `/api/health` | Health check |

---

## Quickstart

Requires **Python 3.10+**.

```bash
git clone https://github.com/devisricharan1250-stack/RAG-Chatbot.git
cd RAG-Chatbot/backend

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # first run downloads the embedding model (~80 MB)

cp .env.example .env               # then add your ANTHROPIC_API_KEY
python main.py
```

Open **http://127.0.0.1:8000**. API docs are at **http://127.0.0.1:8000/docs**.

### Use a different model

```env
# OpenAI (also run: pip install openai)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-key

# Fully local and free with Ollama (no API key; nothing leaves your machine)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
```

### Tuning

| Setting | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 chars | Smaller chunks give more precise retrieval but less context per hit |
| `TOP_K` | 4 | More chunks give more context, but also more noise and cost |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Use `all-mpnet-base-v2` for higher quality at lower speed |

---

## Roadmap

- [ ] Page-level source attribution (document name and page number)
- [ ] Streaming responses
- [ ] Conversation memory for follow-up questions
- [ ] Hybrid search (BM25 + dense) with a reranker
- [ ] Retrieval evaluation (recall@k, answer faithfulness) on a labelled question set
- [ ] Hosted vector DB option (Qdrant / Pinecone) and per-user collections

---

## Tech stack

Python · FastAPI · ChromaDB · Sentence-Transformers · Claude API · OpenAI / Ollama (optional) · HTML/JS
