"""
Central configuration. Everything that you might want to change lives here,
and can be overridden with environment variables (see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads a .env file if present


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ----- Which LLM provider generates the answers -----
# Options: "anthropic" (default), "openai", "ollama"
LLM_PROVIDER = _get("LLM_PROVIDER", "anthropic")

# Model name for the chosen provider.
# Anthropic: e.g. "claude-sonnet-4-6"  (check docs.claude.com for current names)
# OpenAI:    e.g. "gpt-4o-mini"
# Ollama:    e.g. "llama3.1"  (must be pulled locally first: `ollama pull llama3.1`)
LLM_MODEL = _get("LLM_MODEL", "claude-sonnet-4-6")

ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
OLLAMA_HOST = _get("OLLAMA_HOST", "http://localhost:11434")

# ----- Embeddings (runs locally, free) -----
# A small, fast, good-quality sentence embedding model.
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ----- Vector store -----
CHROMA_DIR = _get("CHROMA_DIR", "./chroma_db")     # where the index is persisted
COLLECTION_NAME = _get("COLLECTION_NAME", "documents")

# ----- Chunking -----
CHUNK_SIZE = int(_get("CHUNK_SIZE", "800"))        # characters per chunk
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "150"))  # characters shared between neighbours

# ----- Retrieval -----
TOP_K = int(_get("TOP_K", "4"))                    # how many chunks to feed the LLM

# ----- File uploads -----
UPLOAD_DIR = _get("UPLOAD_DIR", "./uploads")
