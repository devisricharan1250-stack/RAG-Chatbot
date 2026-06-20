"""
The "answering" half of RAG.

Given a question:
  1. retrieve the most relevant chunks from the vector store
  2. stuff them into a prompt that tells the LLM to answer ONLY from them
  3. return the answer together with the sources it was based on
"""
from typing import Dict

import llm
import config
from ingest import DocumentStore


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided "
    "context from the user's documents. If the answer is not contained in the "
    "context, say you don't know based on the documents -- do not invent facts. "
    "Cite the sources you used by their [n] number."
)


def _build_prompt(question: str, chunks) -> str:
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        context_blocks.append(f"[{i}] (from {c['source']}):\n{c['text']}")
    context = "\n\n".join(context_blocks)

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, and reference sources like [1], [2]."
    )


class RAGPipeline:
    def __init__(self, store: DocumentStore):
        self.store = store

    def answer(self, question: str) -> Dict:
        chunks = self.store.search(question, config.TOP_K)

        if not chunks:
            return {
                "answer": "No documents have been indexed yet, so I can't answer "
                          "from your sources. Upload a document first.",
                "sources": [],
            }

        prompt = _build_prompt(question, chunks)
        answer_text = llm.generate(SYSTEM_PROMPT, prompt)

        # Trim chunk text for the UI so we don't ship huge payloads.
        sources = [
            {
                "n": i + 1,
                "source": c["source"],
                "score": c["score"],
                "preview": (c["text"][:240] + "...") if len(c["text"]) > 240 else c["text"],
            }
            for i, c in enumerate(chunks)
        ]
        return {"answer": answer_text, "sources": sources}
