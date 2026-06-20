"""
The "indexing" half of RAG.

Pipeline:  file  ->  raw text  ->  overlapping chunks  ->  embeddings  ->  Chroma

`DocumentStore` owns the embedding model and the vector database, and exposes
two operations the rest of the app needs: add_document() and search().
"""
import os
import uuid
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

import config


# ---------------------------------------------------------------------------
# 1. Loading: turn a file on disk into a single string of text
# ---------------------------------------------------------------------------
def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if ext == ".docx":
        import docx
        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs)

    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    raise ValueError(f"Unsupported file type: {ext}. Use pdf, docx, txt or md.")


# ---------------------------------------------------------------------------
# 2. Chunking: split long text into overlapping windows
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split on character windows with overlap so a sentence cut in half still
    appears whole in a neighbouring chunk. We try to break on a space near the
    boundary so we don't slice words apart."""
    text = " ".join(text.split())  # collapse whitespace/newlines
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        # nudge the cut to the nearest space before `end` for cleaner chunks
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        start = end - overlap  # step back by the overlap
        if start <= 0:
            start = end
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# 3. The store: embeddings + Chroma vector database
# ---------------------------------------------------------------------------
class DocumentStore:
    def __init__(self):
        # Loads the embedding model into memory (downloads once on first run).
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        # Persistent on-disk vector DB so the index survives restarts.
        self.client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def add_document(self, path: str, filename: str) -> int:
        """Load, chunk, embed and store one document. Returns chunk count."""
        text = load_text(path)
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks:
            return 0

        embeddings = self._embed(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query: str, k: int) -> List[Dict]:
        """Return the k most relevant chunks for a question."""
        if self.collection.count() == 0:
            return []

        query_embedding = self._embed([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
        )
        hits = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk": meta.get("chunk", -1),
                "score": round(1 - dist, 3),  # cosine distance -> similarity
            })
        return hits

    def stats(self) -> Dict:
        return {"chunks": self.collection.count()}

    def reset(self):
        """Wipe the whole index (useful when you want to start over)."""
        self.client.delete_collection(config.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
