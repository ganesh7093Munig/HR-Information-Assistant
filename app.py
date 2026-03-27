"""Streamlit entry point for the local HR Policy RAG Assistant."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DB_DIR,
    DEFAULT_FALLBACK_RESPONSE,
    ensure_directory,
    unique_sources_from_documents,
)
from embeddings import DEFAULT_EMBEDDING_MODEL, get_embeddings
from ingestion import load_pdfs_from_folder, split_page_records
from rag_pipeline import (
    DEFAULT_LLM_MODEL,
    build_rag_graph,
    create_retriever,
    get_vectorstore,
    ingest_documents,
    is_vectorstore_empty,
)

# Set the target folder for temporary file uploads
UPLOAD_DIR = Path("data/uploads")


def inject_styles() -> None:
    """Inject minimal custom styles for header and response metadata."""
    # Applying some CSS styling to the Streamlit app to make the header sticky and look better
    st.markdown(
        """
        <style>
        .app-header {
            position: sticky;
            top: 0;
            z-index: 10;
            background-color: var(--background-color, rgb(14, 17, 23));
            border: 1px solid rgba(250, 250, 250, 0.08);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin: 0 0 0.9rem 0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }
        .app-header h1 {
            margin: 0;
            padding: 0;
        }
        .app-header p {
            margin: 0.35rem 0 0 0;
            opacity: 0.9;
        }
        .response-time {
            font-size: 0.82rem;
            opacity: 0.75;
            margin-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render a dedicated header container above the chat area."""
    # Utilizing the custom CSS class 'app-header' to display the main title
    header_container = st.container()
    with header_container:
        st.markdown(
            """
            <div class="app-header">
                <h1>HR Policy RAG Assistant</h1>
                <p>Upload HR policy PDF documents, build a local Chroma index, and ask grounded questions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def init_session_state() -> None:
    """Initialize Streamlit session state keys used by the app."""
    # Set default values for session state variables if they don't exist yet
    defaults = {
        "chat_history": [],                     # Stores the Q&A conversation
        "index_ready": False,                   # Flag indicating if ChromaDB is populated
        "uploaded_file_names": [],              # List of files successfully ingested
        "chunk_size": 900,                      # Size of text chunks for the vector DB
        "chunk_overlap": 150,                   # Overlap between consecutive chunks
        "top_k": 4,                             # Number of documents to retrieve per query
        "collection_name": DEFAULT_COLLECTION_NAME, # Chroma collection name
        "active_collection_name": DEFAULT_COLLECTION_NAME, # Tracks changes to collection name
        "db_dir": str(DEFAULT_DB_DIR),          # Location of the Chroma database on disk
        "index_status_checked": False,          # Make sure we only check index status once on load
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def prepare_upload_dir(upload_dir: Path) -> Path:
    """Reset and recreate the temporary upload directory."""
    # Delete the entire uploads directory if it exists to clear out old files
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    # Recreate the clean directory structure
    return ensure_directory(upload_dir)


def save_uploaded_files(uploaded_files, upload_dir: Path) -> list[Path]:
    """Save uploaded Streamlit files with duplicate-safe names."""
    saved_paths: list[Path] = []
    # Loop over all files uploaded by the user
    for file_obj in uploaded_files:
        original_name = Path(file_obj.name).name
        stem = Path(original_name).stem
        suffix = Path(original_name).suffix or ".pdf"

        # Determine target file path
        target = upload_dir / f"{stem}{suffix}"
        counter = 1
        
        # If a file with the same name already exists, append a counter to the filename
        while target.exists():
            target = upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        # Write the binary data from the uploaded file buffer to disk
        target.write_bytes(file_obj.getbuffer())
        saved_paths.append(target)

    return saved_paths


def build_index_from_uploads(
    uploaded_files,
    chunk_size: int,
    chunk_overlap: int,
    db_dir: Path,
    collection_name: str,
) -> dict[str, int]:
    """Build or rebuild a Chroma index from uploaded PDFs."""
    # Validate we actually have files to process
    if not uploaded_files:
        raise ValueError("Upload at least one PDF file before building the index.")

    # Prepare temp directory and save files
    upload_dir = prepare_upload_dir(UPLOAD_DIR)
    saved_paths = save_uploaded_files(uploaded_files, upload_dir)

    # Extract raw text from the PDFs
    records = load_pdfs_from_folder(upload_dir)
    if not records:
        raise RuntimeError("No readable text was extracted from uploaded PDF files.")

    # Split the raw text into manageable chunks
    chunks = split_page_records(
        records=records,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("No chunks were created from uploaded PDF files.")

    # Setup the embedding model
    embeddings = get_embeddings(model=DEFAULT_EMBEDDING_MODEL)
    
    # Store the chunked vectors into ChromaDB (resetting the collection first)
    ingest_documents(
        documents=chunks,
        persist_directory=db_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
        reset=True,
    )

    # Save the updated list of file names in the session state
    st.session_state.uploaded_file_names = [path.name for path in saved_paths]
    
    # Return processing stats
    return {
        "file_count": len(saved_paths),
        "page_count": len(records),
        "chunk_count": len(chunks),
    }


def answer_question(question: str, top_k: int, db_dir: Path, collection_name: str):
    """Run retrieval and grounded answer generation for one user question."""
    embeddings = get_embeddings(model=DEFAULT_EMBEDDING_MODEL)
    # Prepare the vector store connection
    vectorstore = get_vectorstore(
        persist_directory=db_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    
    # Check if we have data to query against
    if is_vectorstore_empty(vectorstore):
        raise RuntimeError("The index is empty. Build the index from uploaded PDFs first.")

    # Setup retriever to get top K documents
    retriever = create_retriever(vectorstore=vectorstore, k=top_k)
    # Build LangGraph workflow connecting retriever and LLM
    rag_graph = build_rag_graph(retriever=retriever)
    
    # Run the workflow with the user question
    result = rag_graph.invoke({"question": question})

    # Extract the results
    documents = list(result.get("documents", []))
    answer = str(result.get("answer", DEFAULT_FALLBACK_RESPONSE)).strip()
    
    # Ensure answer makes sense
    if not answer or not documents:
        answer = DEFAULT_FALLBACK_RESPONSE

    # Extract unique sources for citations
    sources = unique_sources_from_documents(documents)
    return answer, sources


def render_sidebar():
    """Render sidebar controls and return uploaded files."""
    with st.sidebar:
        st.subheader("Configuration")
        # Display the active model configurations
        st.markdown(f"- LLM: `{DEFAULT_LLM_MODEL}`")
        st.markdown(f"- Embeddings: `{DEFAULT_EMBEDDING_MODEL}`")
        st.markdown("- Vector DB: `Chroma`")

        # The file uploader widget
        uploaded_files = st.file_uploader(
            "Upload HR policy PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to build the retrieval index.",
        )

        # Show currently uploaded files info
        if uploaded_files:
            st.success(f"{len(uploaded_files)} PDF file(s) selected.")
            st.session_state.uploaded_file_names = [
                Path(file_obj.name).name for file_obj in uploaded_files
            ]
            with st.expander("Selected files", expanded=False):
                for file_name in st.session_state.uploaded_file_names:
                    st.write(f"- {file_name}")
        else:
            st.info("Upload files, then click Build / Rebuild Index.")

        # Inputs for chunk size, overlap, and top_k
        st.number_input(
            "Chunk size",
            min_value=300,
            max_value=2000,
            step=50,
            key="chunk_size",
        )
        st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=500,
            step=10,
            key="chunk_overlap",
        )
        st.number_input(
            "Top-k retrieval",
            min_value=1,
            max_value=10,
            step=1,
            key="top_k",
        )
        
        # Input for collection name, triggering resets if changed
        st.text_input("Collection name", key="collection_name")
        if st.session_state.collection_name != st.session_state.active_collection_name:
            st.session_state.active_collection_name = st.session_state.collection_name
            st.session_state.index_ready = False
            st.session_state.index_status_checked = False

        # Action buttons
        build_clicked = st.button("Build / Rebuild Index", type="primary", use_container_width=True)
        clear_chat_clicked = st.button("Clear Chat", use_container_width=True)

        # Logic for clearing chat
        if clear_chat_clicked:
            st.session_state.chat_history = []
            st.success("Chat history cleared.")

        # Logic for index building
        if build_clicked:
            with st.spinner("Building index from uploaded files..."):
                try:
                    # Execute the vector ingestion
                    stats = build_index_from_uploads(
                        uploaded_files=uploaded_files,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap,
                        db_dir=Path(st.session_state.db_dir),
                        collection_name=st.session_state.collection_name,
                    )
                except Exception as exc:
                    st.session_state.index_ready = False
                    st.error(f"Index build failed: {exc}")
                else:
                    st.session_state.index_ready = True
                    st.success(
                        "Index built successfully "
                        f"({stats['file_count']} files, {stats['page_count']} pages, {stats['chunk_count']} chunks)."
                    )

        # Show current readiness of the vector store
        if st.session_state.index_ready:
            st.success("Index status: Ready")
        else:
            st.warning("Index status: Not ready")

    return uploaded_files


def render_chat() -> None:
    """Render chat history and handle user questions."""
    chat_container = st.container()
    
    # Loop over past history and display messages
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                sources = message.get("sources") or []
                response_time = message.get("response_time")
                
                # Render sources nicely below assistant answers
                if message["role"] == "assistant" and sources:
                    st.markdown("**Sources**")
                    for source in sources:
                        st.markdown(f"- {source}")
                        
                # Display response latency if it's an assistant message
                if message["role"] == "assistant" and response_time is not None:
                    st.markdown(
                        f"<div class='response-time'>Response time: {response_time:.2f} sec</div>",
                        unsafe_allow_html=True,
                    )

    # Input for new user question
    user_question = st.chat_input("Ask a question about the uploaded HR policy PDFs")
    if not user_question:
        return

    # Add question to state and display locally immediately
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Process assistant response mechanism
    with st.chat_message("assistant"):
        response_time: float | None = None
        answer = ""
        sources: list[str] = []

        # Enforce that index is ready before searching
        if not st.session_state.index_ready:
            answer = "Please upload PDF files and click Build / Rebuild Index first."
            st.warning(answer)
        else:
            with st.spinner("Retrieving policy context and generating answer..."):
                started_at = time.perf_counter()
                try:
                    # Run RAG execution
                    answer, sources = answer_question(
                        question=user_question,
                        top_k=st.session_state.top_k,
                        db_dir=Path(st.session_state.db_dir),
                        collection_name=st.session_state.collection_name,
                    )
                except Exception as exc:
                    # Trap errors securely and show to user
                    answer = f"I ran into an error while processing your question: {exc}"
                    st.error(answer)
                finally:
                    # Calculate execution duration
                    response_time = time.perf_counter() - started_at

            # Print main block if no major exceptions
            if answer and not answer.startswith("I ran into an error while processing your question:"):
                st.markdown(answer)
                if sources:
                    st.markdown("**Sources**")
                    for source in sources:
                        st.markdown(f"- {source}")
                else:
                    st.caption("No relevant source chunks were retrieved.")

        # Show performance metadata tracking execution speed
        if response_time is not None:
            st.markdown(
                f"<div class='response-time'>Response time: {response_time:.2f} sec</div>",
                unsafe_allow_html=True,
            )

        # Append generated answer to global state loop
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "response_time": response_time,
            }
        )


def main() -> None:
    """Run the Streamlit app."""
    # Ensure environment variables are loaded securely
    load_dotenv()
    # Configure main tab and initial loading params
    st.set_page_config(page_title="HR Policy RAG Assistant", layout="wide")
    
    # Initialize basic configurations
    inject_styles()
    init_session_state()
    
    # Do an initial check if ChromaDB has data upon app starting
    if not st.session_state.index_status_checked:
        try:
            embeddings = get_embeddings(model=DEFAULT_EMBEDDING_MODEL)
            vectorstore = get_vectorstore(
                persist_directory=Path(st.session_state.db_dir),
                collection_name=st.session_state.collection_name,
                embedding_function=embeddings,
            )
            st.session_state.index_ready = not is_vectorstore_empty(vectorstore)
        except Exception:
            st.session_state.index_ready = False
        finally:
            st.session_state.index_status_checked = True

    # Render main components
    render_header()
    render_sidebar()
    render_chat()

# Bootstrap entry point into Streamlit
if __name__ == "__main__":
    main()
