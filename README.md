# Margin — a RAG chat application (built from scratch)

Upload your documents, then ask questions and get answers grounded **only** in
those documents, with sources. This repo contains the three pieces you asked for:
a **frontend**, a **backend**, and the **RAG pipeline** itself.

---

## 1. What is RAG, in one minute

A plain language model answers from memory and can make things up. **RAG
(Retrieval-Augmented Generation)** fixes that by giving the model your documents
to read *at question time*. It works in two phases:

**Indexing (done once per document, when you upload):**

```
file → extract text → split into chunks → turn each chunk into a vector
                                          → store vectors in a database
```

**Answering (done for every question):**

```
question → turn into a vector → find the most similar chunks
        → put those chunks in a prompt → LLM writes an answer from them
```

A "vector" (or *embedding*) is just a list of numbers that captures meaning, so
two pieces of text about the same thing land near each other. Finding relevant
chunks = finding the nearest vectors.

---

## 2. The stack (and why)

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Free, runs locally, no API needed |
| Vector DB | ChromaDB | Free, local, persists to disk, simple API |
| Answer model | Claude API (swappable) | Best answer quality with near-zero setup |
| Backend | FastAPI | Small, fast, automatic docs at `/docs` |
| Frontend | One `index.html` | No build step, easy to read and change |

The answer model is **swappable** — flip `LLM_PROVIDER` to `openai` or `ollama`
(fully free + private) without touching any other code.

---

## 3. Project layout

```
rag-chat-app/
├── backend/
│   ├── config.py        # all settings (env-overridable)
│   ├── llm.py           # the swappable answer model (anthropic/openai/ollama)
│   ├── ingest.py        # load → chunk → embed → store, and search
│   ├── rag.py           # retrieve + build prompt + generate
│   ├── main.py          # FastAPI server (the HTTP endpoints)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html       # the whole UI (chat + upload + sources)
└── README.md
```

### What each backend file does

- **`config.py`** — one place for every setting. Reads from a `.env` file so you
  never hard-code secrets.
- **`ingest.py`** — the indexing half. `load_text()` reads PDF/DOCX/TXT/MD;
  `chunk_text()` splits text into overlapping windows; `DocumentStore` holds the
  embedding model + Chroma and exposes `add_document()` and `search()`.
- **`rag.py`** — the answering half. `RAGPipeline.answer()` searches for the top
  chunks, builds a grounded prompt (with a system instruction that forbids making
  things up), calls the LLM, and returns the answer plus the sources used.
- **`llm.py`** — isolates the model call so the rest of the app is
  provider-agnostic.
- **`main.py`** — exposes `/api/upload`, `/api/chat`, `/api/reset`, `/api/health`
  and serves the frontend.

---

## 4. Setup

You need **Python 3.10+**.

```bash
cd rag-chat-app/backend

# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies (first run downloads the embedding model, ~80 MB)
pip install -r requirements.txt

# 3. configure
cp .env.example .env
#   then open .env and paste your ANTHROPIC_API_KEY
#   (get one at https://console.anthropic.com)
```

### Run it

```bash
python main.py
```

Open **http://127.0.0.1:8000** in your browser. The server also serves the
frontend, so that's the only command you need. (You can also open
`frontend/index.html` directly — it will call the server at `127.0.0.1:8000`.)

Interactive API docs are at **http://127.0.0.1:8000/docs**.

---

## 5. Using it

1. Click **Add document** (or drag a file onto the page). Wait for the
   "X chunks" confirmation.
2. Type a question and press **Ask**.
3. The answer appears with `[1] [2]` citation chips, and the **Sources** rail on
   the right shows the exact chunks it used, with a similarity score.
4. **Clear index** wipes everything to start fresh.

---

## 6. Switching the answer model

Edit `.env`:

**OpenAI**
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-key
```
(then `pip install openai`)

**Fully local & free with Ollama** — no API key, nothing leaves your machine:
```
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
```
First install Ollama (ollama.com), then `ollama pull llama3.1` and make sure
`ollama serve` is running.

---

## 7. Tuning quality

All in `.env` / `config.py`:

- **`CHUNK_SIZE` / `CHUNK_OVERLAP`** — smaller chunks = more precise retrieval but
  less context per chunk. 800 / 150 chars is a solid default.
- **`TOP_K`** — how many chunks to feed the model. More = more context but more
  noise and cost. 4 is a good start.
- **`EMBEDDING_MODEL`** — `all-MiniLM-L6-v2` is fast; try `all-mpnet-base-v2` for
  higher quality (slower).

---

## 8. Troubleshooting

- **"ANTHROPIC_API_KEY is not set"** — you didn't fill in `.env`, or didn't copy
  it from `.env.example`.
- **First request is slow** — the embedding model downloads and loads once; later
  runs are fast.
- **Answer says "I don't know based on the documents"** — that's the model being
  honest because the answer wasn't in your chunks. Try rephrasing, raising
  `TOP_K`, or adding the relevant document.
- **Frontend can't reach the backend** — make sure the server is running on port
  8000, or just open `http://127.0.0.1:8000` directly.

---

## 9. Where to take it next

- Show which document + page a chunk came from (add page numbers in `load_text`).
- Stream the answer token-by-token for a live "typing" feel.
- Add conversation memory so follow-up questions understand "it"/"that".
- Swap Chroma for a hosted vector DB (Pinecone, Qdrant) when you outgrow local.
- Add user accounts and per-user collections.

---

## 10. The end-to-end flow, one more time

```
You upload report.pdf
   → backend extracts text, makes ~40 overlapping chunks
   → each chunk → 384-number vector → stored in Chroma

You ask "What were Q3 revenues?"
   → question → vector → Chroma returns the 4 closest chunks
   → those chunks + your question → Claude
   → Claude answers using only them, citing [1]..[4]
   → you see the answer + the source chunks it leaned on
```

That's RAG.
