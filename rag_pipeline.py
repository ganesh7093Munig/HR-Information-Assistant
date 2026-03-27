"""LangGraph workflow, retriever, and vector store setup for the HR policy RAG project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_FALLBACK_RESPONSE,
    DEFAULT_DB_DIR,
    DEFAULT_COLLECTION_NAME,
    ensure_directory,
    get_ollama_base_url,
)
from embeddings import get_embeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma


# --- Vector Store ---

def get_vectorstore(
    persist_directory: str | Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_function=None,
) -> Chroma:
    """Create or load a persistent Chroma vector store."""
    # Ensure the directory where DB will be saved exists
    db_path = ensure_directory(persist_directory)
    # Initialize embedding generation, default to our local Ollama model
    embeddings = embedding_function or get_embeddings()
    # Initialize and return Chroma DB client
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(db_path),
        embedding_function=embeddings,
    )


def get_vector_count(vectorstore: Chroma) -> int:
    """Return total indexed vectors in the Chroma collection."""
    try:
        # Protected internal call to Chroma to grab the number of saved vectors
        return int(vectorstore._collection.count())  # noqa: SLF001
    except Exception:
        # Return 0 on error
        return 0


def is_vectorstore_empty(vectorstore: Chroma) -> bool:
    """Return True when the collection has no vectors."""
    # Helper to check if the ChromaDB needs to be populated
    return get_vector_count(vectorstore) == 0


def ingest_documents(
    documents: Sequence[Document],
    persist_directory: str | Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_function=None,
    reset: bool = False,
) -> Chroma:
    """Ingest chunked documents into persistent Chroma storage."""
    # Ensure that we actually have documents to parse
    if not documents:
        raise ValueError("No documents were provided for ingestion.")

    # Get absolute path for persistence
    db_path = Path(persist_directory).expanduser()
    
    # Establish connection with the vector database
    vectorstore = get_vectorstore(
        persist_directory=db_path,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )

    # Optional reset triggered when updating the entire dataset
    if reset:
        try:
            # Drop existing collection and all vectors
            if hasattr(vectorstore, "delete_collection"):
                vectorstore.delete_collection()
            # Recreate an empty database in the same location
            vectorstore = get_vectorstore(
                persist_directory=db_path,
                collection_name=collection_name,
                embedding_function=embedding_function,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to reset Chroma collection. "
                "Close other running app instances and try again."
            ) from exc

    # Grab previously established chunk_ids for idempotent inserts
    ids: list[str] = []
    for idx, doc in enumerate(documents):
        # We try to use the deterministic hash we generated during chunking
        chunk_id = str(doc.metadata.get("chunk_id", f"doc-{idx}"))
        ids.append(chunk_id)

    # Bulk insert all parsed Text Document chunks and their IDs
    vectorstore.add_documents(list(documents), ids=ids)
    
    # Force saving to disk if the API supports it
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    return vectorstore


# --- Retriever ---

def create_retriever(
    vectorstore,
    k: int = 4,
    search_type: str = "similarity",
) -> BaseRetriever:
    """Create a retriever from a Chroma vectorstore."""
    # The retriever acts as the search engine for LangChain, returning the top K nearest results
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


def retrieve_documents(retriever: BaseRetriever, query: str) -> list[Document]:
    """Retrieve top-k relevant documents for a query."""
    # Pass user query string directly into vectorstore search
    results = retriever.invoke(query)
    # Guarantee we return a list, empty if necessary
    return list(results or [])


# --- LangGraph RAG Workflow ---

# Create prompt format matching how we want local LLMs to process queries
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an HR policy assistant. "
                "Answer only using the provided context. "
                "If the answer is not explicitly present in the context, "
                "reply exactly with: {fallback_response}"
            ),
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}",
        ),
    ]
)


class RAGState(TypedDict, total=False):
    """State passed through the LangGraph workflow."""
    # Represents the variables that are passed between the LangGraph nodes
    question: str
    documents: list[Document]
    context: str
    answer: str


def _render_context(documents: list[Document]) -> str:
    """Render retrieved documents into a single context string."""
    context_blocks: list[str] = []
    # Join document blocks along with source hints for the LLM to provide citations
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        context_blocks.append(
            f"[Source: {source}, Page: {page}]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(context_blocks).strip()


def build_llm(
    model: str = DEFAULT_LLM_MODEL, base_url: str | None = None
) -> ChatOllama:
    """Create the Ollama chat model used for answer generation."""
    # Fallback to local machine url if one wasn't explicitly given
    resolved_base_url = base_url or get_ollama_base_url()
    # Provide Chat API wrapper around the active Ollama model, temperature 0 for factual accuracy
    return ChatOllama(model=model, base_url=resolved_base_url, temperature=0)


def build_rag_graph(
    retriever,
    llm: ChatOllama | None = None,
    fallback_response: str = DEFAULT_FALLBACK_RESPONSE,
):
    """Build and compile a simple retrieve -> answer LangGraph flow."""
    model = llm or build_llm()

    def retrieve_node(state: RAGState) -> RAGState:
        # First step in graph: grab question and lookup documents
        question = state.get("question", "").strip()
        if not question:
            # Handle empty query case gracefully
            return {"documents": [], "context": ""}

        # Do vector search
        docs = retrieve_documents(retriever, question)
        # Format list of documents into a single block of text for LLM prompt insertion
        context = _render_context(docs)
        # Add updated variables to global state
        return {"documents": docs, "context": context}

    def answer_node(state: RAGState) -> RAGState:
        # Second step in graph: send text to local LLM for generation
        question = state.get("question", "").strip()
        context = state.get("context", "").strip()
        docs = state.get("documents", [])

        # Abort if data is missing or query invalid
        if not question or not context or not docs:
            return {"answer": fallback_response}

        # Build prompt using user question and retrieved vector data
        prompt_value = RAG_PROMPT.invoke(
            {
                "question": question,
                "context": context,
                "fallback_response": fallback_response,
            }
        )
        
        # Stream prompt to Ollama model synchronously
        response = model.invoke(prompt_value)
        # Extract returned string
        answer = str(getattr(response, "content", "")).strip()
        
        # Inject standard fallback phrase if generation fails completely
        if not answer:
            answer = fallback_response
            
        return {"answer": answer}

    # Set up our explicit node topology
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer", answer_node)
    
    # Establish entry point
    workflow.set_entry_point("retrieve")
    
    # Establish connection path between nodes
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", END)
    
    # Process edges and nodes into executable
    return workflow.compile()
