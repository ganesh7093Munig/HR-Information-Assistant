"""Embedding model initialization for the RAG pipeline."""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from config import DEFAULT_EMBEDDING_MODEL, get_ollama_base_url


def get_embeddings(
    model: str = DEFAULT_EMBEDDING_MODEL, base_url: str | None = None
) -> OllamaEmbeddings:
    """Return an Ollama embedding client for local embedding generation."""
    # Resolve the base URL, falling back to config defaults
    resolved_base_url = base_url or get_ollama_base_url()
    # Initialize and return the Langchain OllamaEmbeddings object
    return OllamaEmbeddings(model=model, base_url=resolved_base_url)


def check_embeddings_ready(embeddings: OllamaEmbeddings) -> bool:
    """Quick connectivity check against the embedding model."""
    try:
        # Attempt to embed a tiny string to verify the model is loaded and reachable
        _ = embeddings.embed_query("health-check")
        return True
    except Exception:
        # If any error occurs (e.g., Ollama is down or model is missing), return False
        return False
