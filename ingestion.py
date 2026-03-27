"""PDF loading, text extraction, and chunking for the HR policy RAG project."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import fitz
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import PageRecord, is_pdf_file, normalize_text, build_chunk_id

logger = logging.getLogger(__name__)


# --- PDF Loading ---

def load_pdf_pages(pdf_path: str | Path) -> list[PageRecord]:
    """Load a single PDF and return page-level extracted text records."""
    # Ensure the path is evaluated properly
    path = Path(pdf_path).expanduser()
    
    # Check if the file actually exists
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
        
    # Check that it's a file and has a .pdf extension
    if not path.is_file() or not is_pdf_file(path):
        raise ValueError(f"Expected a PDF file, got: {path}")

    records: list[PageRecord] = []
    
    # Try opening the PDF Document using PyMuPDF (fitz)
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{path}': {exc}") from exc

    # Use context manager to ensure the document is closed automatically
    with doc:
        # Iterate over all pages (1-indexed for biological readability)
        for page_idx, page in enumerate(doc, start=1):
            try:
                # Extract raw text and normalize the whitespace
                text = normalize_text(page.get_text("text"))
            except Exception:
                # Log an error if text extraction fails for a specific page, but continue to next
                logger.exception(
                    "Failed to read text from %s page %s", path.name, page_idx
                )
                continue

            # Skip empty pages
            if not text:
                continue

            # Store the extracted text and its metadata location into a PageRecord
            records.append(
                PageRecord(text=text, page_number=page_idx, source_file=path.name)
            )

    return records


def iter_pdf_files(folder_path: str | Path, recursive: bool = False) -> Iterable[Path]:
    """Yield PDF files from a folder in deterministic order."""
    folder = Path(folder_path).expanduser()
    # Support searching in subdirectories if recursive=True
    pattern = "**/*.pdf" if recursive else "*.pdf"
    # Sort files alphabetically for consistency yielding a deterministic list
    yield from sorted(folder.glob(pattern))


def load_pdfs_from_folder(
    folder_path: str | Path, recursive: bool = False
) -> list[PageRecord]:
    """Load all PDFs from a folder and return combined page records."""
    folder = Path(folder_path).expanduser()
    
    # Verify the target is a valid, existing directory
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Expected a directory path, got: {folder}")

    records: list[PageRecord] = []
    # Collect all PDF files in the directory
    pdf_files = list(iter_pdf_files(folder, recursive=recursive))
    
    if not pdf_files:
        logger.warning("No PDF files found in folder: %s", folder)
        return records

    # Iterate through all discovered PDFs
    for pdf_file in pdf_files:
        try:
            # Extract pages from this PDF and append to main list
            records.extend(load_pdf_pages(pdf_file))
        except Exception:
            # If one file is corrupt, log and skip rather than crashing the whole process
            logger.exception("Skipping unreadable PDF: %s", pdf_file)

    return records


# --- Text Splitting ---

def get_text_splitter(
    chunk_size: int = 900, chunk_overlap: int = 150
) -> RecursiveCharacterTextSplitter:
    """Create the default recursive character splitter."""
    # This splitter splits based on paragraphs, then sentences, then words
    # keeping related text grouped together while meeting the size constraint
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def page_records_to_documents(records: Iterable[PageRecord]) -> list[Document]:
    """Convert page records into LangChain Document objects."""
    documents: list[Document] = []
    for record in records:
        # Ignore completely whitespace pages
        if not record.text.strip():
            continue
        # Standardize our PageRecord to Langchain Document format required by Chroma
        documents.append(
            Document(page_content=record.text, metadata=record.to_metadata())
        )
    return documents


def split_documents(
    documents: Sequence[Document],
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[Document]:
    """Split documents while preserving and extending metadata."""
    # Setup our chunker
    active_splitter = splitter or get_text_splitter()
    
    # Split the large page-level documents into smaller chunks
    chunks = active_splitter.split_documents(list(documents))

    # Keep track of how many chunks each page was split into to assign index numbers
    chunk_counters: dict[tuple[str, int], int] = defaultdict(int)
    
    # Add robust metadata and IDs to each generated chunk
    for chunk in chunks:
        metadata = chunk.metadata or {}
        source = str(metadata.get("source", "unknown"))
        # Page defaults to -1 if we can't find it for some reason
        page = int(metadata.get("page", -1))

        key = (source, page)
        
        # Get current index for this specific page, then inc for the next chunk
        chunk_index = chunk_counters[key]
        chunk_counters[key] += 1

        chunk.metadata["chunk_index"] = chunk_index
        # Build unique stable identifier for the ChromaDB store
        chunk.metadata["chunk_id"] = build_chunk_id(
            source=source,
            page=page,
            chunk_index=chunk_index,
            text=chunk.page_content,
        )

    return chunks


def split_page_records(
    records: Iterable[PageRecord],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Convert page records to chunked documents in one step."""
    # Setup length rules
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Convert local representation (PageRecord) to LangChain form (Document)
    page_docs = page_records_to_documents(records)
    # Slice the documents into bite-sized pieces for vector embedding
    return split_documents(page_docs, splitter=splitter)
