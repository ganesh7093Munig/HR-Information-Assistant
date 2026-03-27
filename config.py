"""Shared configuration and helpers for the HR policy RAG project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

# --- Constants ---

# The embedding model used by Ollama to convert text to vectors
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
# The language model used by Ollama to answer questions
DEFAULT_LLM_MODEL = "qwen2.5:7b"
# The base URL where the local Ollama service is running
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
# The name of the collection in ChromaDB where embeddings are stored
DEFAULT_COLLECTION_NAME = "hr-policy-index"
# The default directory where the local ChromaDB database will be persisted
DEFAULT_DB_DIR = Path("chroma_db")
# The response provided if the RAG system cannot find an answer in the source text
DEFAULT_FALLBACK_RESPONSE = (
    "I could not find this information in the uploaded HR policy documents."
)


# --- Data Structures ---

@dataclass(frozen=True)
class PageRecord:
    """Represents extracted text for one PDF page."""

    # The actual text content extracted from the page
    text: str
    # The page number of the PDF
    page_number: int
    # The name of the PDF file from which this was extracted
    source_file: str

    def to_metadata(self) -> dict[str, Any]:
        """Convert the page record to metadata used by chunked documents."""
        # This metadata helps link the chunk back to its source file and page
        return {"source": self.source_file, "page": self.page_number}


# --- Helpers ---

def ensure_directory(path: str | Path) -> Path:
    """Create directory when missing and return a resolved path."""
    # Convert string or relative path to an absolute/resolved Path object
    directory = Path(path).expanduser()
    # Create the directory structure, avoiding errors if it already exists
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_pdf_file(path: str | Path) -> bool:
    """Return True when the file extension is .pdf."""
    # Check if the file ends with a case-insensitive '.pdf' extension
    return Path(path).suffix.lower() == ".pdf"


def normalize_text(text: str) -> str:
    """Normalize extracted text while preserving paragraph breaks."""
    # Strip trailing and leading whitespace from every line
    lines = [line.strip() for line in text.splitlines()]
    # Re-join non-empty lines with a single newline character
    return "\n".join(line for line in lines if line)


def build_chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    """Build a stable chunk id based on source metadata and content."""
    # Create a unique payload string using file name, page, chunk number, and a snippet of text
    payload = f"{source}|{page}|{chunk_index}|{text[:160]}"
    # Hash the payload to generate a unique, short ID for ChromaDB
    digest = sha1(payload.encode("utf-8")).hexdigest()  # noqa: S324
    return f"{source}-p{page}-c{chunk_index}-{digest[:12]}"


def format_source_reference(metadata: dict[str, Any]) -> str:
    """Create a source label from document metadata."""
    # Extract the source file name from the metadata dict
    source = str(metadata.get("source", "unknown"))
    page = metadata.get("page")
    # If there is no page information, just return the file name
    if page is None:
        return source
    # Formats reference as "filename.pdf - page X"
    return f"{source} - page {page}"


def unique_sources_from_documents(documents: Iterable[Any]) -> list[str]:
    """Return unique source labels from LangChain documents."""
    seen: set[str] = set()
    ordered: list[str] = []
    # Loop over all retrieved chunks
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        # Get a readable label for where this chunk came from
        label = format_source_reference(metadata)
        # Keep track of unique labels only while preserving order
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def get_ollama_base_url() -> str:
    """Return OLLAMA_BASE_URL from environment or default."""
    # Look for an environment variable; if not found, use the default port
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
