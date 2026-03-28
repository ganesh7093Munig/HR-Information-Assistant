# HR Policy RAG Assistant

HR Policy RAG Assistant is a local AI-powered document Q&A application built with **Python, Streamlit, LangChain, LangGraph, Ollama, and ChromaDB**. It allows users to upload HR policy PDF documents, build a local searchable knowledge base, and ask natural-language questions to get grounded answers with source references.

## Features

- Upload one or more HR policy PDF files
- Extract PDF content locally
- Split documents into chunks for retrieval
- Generate embeddings using **Ollama** with `nomic-embed-text`
- Store embeddings in **ChromaDB**
- Ask HR-related questions in a chat-style interface
- Generate answers using **Qwen2.5:7b** via Ollama
- Show source references with file name and page number
- Return a fallback response when the answer is not found in the uploaded documents
- Fully local setup with no external API dependency

## Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **LangGraph**
- **Ollama**
- **Qwen2.5:7b**
- **nomic-embed-text**
- **ChromaDB**
- **PyMuPDF**

```bash
HR-Policy-RAG-Assistant/
│
├── app.py
├── config.py
├── embeddings.py
├── ingestion.py
├── rag_pipeline.py
├── requirements.txt
├── README.md
├── data/
│   ├── hr_policy_1.pdf
│   └── hr_policy_2.pdf
└── chroma_db/
```

## How It Works

1. Upload HR policy PDF files
2. Extract text page by page
3. Split text into chunks
4. Generate embeddings for each chunk using `nomic-embed-text`
5. Store chunks and metadata in ChromaDB
6. Retrieve the most relevant chunks for a user question
7. Use `qwen2.5:7b` to generate an answer based only on retrieved context
8. Display the response along with source references

## Architecture & Strategy Deep Dive

The pipeline flow follows standard RAG architecture: **Documents → Chunking → Embeddings → Vector Store → Query → Retrieval → LLM → Response**.
Below is a detailed breakdown of each step and the specific implementation strategies used in the codebase:

### 1. Documents (Ingestion)
- **Strategy:** HR policy PDFs are parsed using PyMuPDF (`fitz`). Text is extracted page-by-page while normalizing whitespace but preserving paragraph breaks. Most crucially, metadata (source filename and page number) is attached immediately to each extracted page record via a `PageRecord` dataclass (`ingestion.py`). This ensures accurate provenance for citations later.

### 2. Chunking
- **Strategy:** Extracted text is split using LangChain's `RecursiveCharacterTextSplitter`. The primary strategy here is to split on natural boundaries (double newlines, single newlines, then periods) to keep distinct semantic ideas together instead of cutting off mid-sentence. We also generate a stable SHA1 hash for each chunk ID based on its content and metadata (`config.py`). This stable deterministic chunk identifier prevents vector duplication and allows idempotent document updates.

### 3. Embeddings
- **Strategy:** Text chunks are transformed into dense vector representations using the `nomic-embed-text` model via local Ollama. The strategy to use `nomic` heavily relies on its optimizations for high-quality semantic search in a local environment. Ensuring all data is embedded locally guarantees that sensitive internal HR documents never leave the local machine architecture (`embeddings.py`).

### 4. Vector Store
- **Strategy:** Embedded chunks are stored locally using **ChromaDB**. It is chosen for its lightweight, file-based persistence (managed in the local `chroma_db/` directory). The strategy is to utilize Chroma for highly efficient nearest-neighbor similarity searches without needing complex infrastructure like a standalone or cloud vector database container (`rag_pipeline.py`).

### 5. Query
- **Strategy:** The query is captured via the Streamlit chat interface. The application processes user inputs sequentially while maintaining Streamlit `session_state` for chat history tracking (`app.py`).

### 6. Retrieval
- **Strategy:** The incoming user question is converted into an embedding using the exact same local `nomic-embed-text` model. Chroma performs a similarity search to fetch the top `K` most contextually relevant chunks. The strategy to retrieve and inject only highly pertinent snippets prevents the LLM's context window from being overloaded with irrelevant noise.

### 7. LLM (Generation)
- **Strategy:** An orchestration workflow is built using **LangGraph**, constructing a state graph passing the context and query between sequential nodes (`rag_pipeline.py`). A highly restricted prompt is passed to the local `qwen2.5:7b` model with the temperature explicitly set to `0`. The strategy forces the model to act extremely deterministically and stay purely factual, relying *only* on the retrieved chunks instead of its pretrained weights, drastically reducing the chance of hallucinating policies.

### 8. Response
- **Strategy:** The LLM output is aggregated and presented in the UI alongside distinct source citations extracted from the chunk metadata (e.g., "filename.pdf - page X"). If the context does not contain the answer, the LLM falls back to a deterministic, configured failure message defined in (`config.py`). The strategy ensures employees are met with transparent "I don't know" answers instead of fabricated or legally incorrect HR guidelines.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://ganesh7093Munig/HR-Policy-RAG-Assistant.git
cd HR-Policy-RAG-Assistant
```

### 2. Create and activate a virtual environment

#### Windows

```bash
python -m venv rag_env
rag_env\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv rag_env
source rag_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama on your machine, then pull the required models:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

## Example Questions

- How many annual leave days are allowed?
- Can employees carry forward unused leave?
- What is the probation period?
- How many remote work days are allowed per week?
- When does private health insurance start?

## Example Use Cases

This project can be adapted for:

- HR policy assistants
- employee handbook Q&A
- internal knowledge assistants
- document-based enterprise chatbots

## Notes

- This project runs fully locally using Ollama
- Best suited for small to medium PDF collections
- Response quality depends on document quality, chunking, retrieval settings, and model performance
- If no relevant information is found, the assistant returns a safe fallback response

## Assumptions & Limitations

### Assumptions
- **Local Environment:** The system assumes the user has a machine capable of running local LLMs and embedding models via Ollama. It relies on decent RAM and CPU/GPU capabilities.
- **Document Structure:** The system assumes that the uploaded HR policies are text-heavy PDFs. Complex diagrams, tables, or scanned images without OCR will not be parsed effectively by PyMuPDF (`fitz`).
- **Language:** The current implementation using `qwen2.5:7b` and `nomic-embed-text` assumes the primary language of the HR policies is English, although Qwen has multilingual capabilities.
- **Idempotency:** When rebuilding the index, it assumes a full wipe of the Chroma DB is acceptable rather than doing intelligent differential updates.

### Limitations (Design Decisions)
- **In-Memory History:** Chat history is maintained only during the active browser session via Streamlit's `session_state`. Refreshing the page clears the history.
- **Sequential Generation:** The system blocks the UI while the local LLM generates answers. There is no streaming of tokens to the UI, sacrificing perceived latency for architectural simplicity.
- **Single Collection:** All documents are embedded into a single Chroma collection. Currently, the system does not support filtering by specific sub-collections or departments.
- **Naive Chunking:** The system uses standard Recursive Character chunking. It might split context midway, which can occasionally cause the retriever to miss nuanced, cross-paragraph context.

## Future Improvements

### 1. Scaling the Knowledge Base
- **Advanced Vector Databases:** Migrate from local ChromaDB to a scalable, managed vector database (like Pinecone, Qdrant, or Milvus) if the document count reaches enterprise scale.
- **Hierarchical Indexing & Filtering:** Implement metadata filtering. Allow users to tag documents by department, location, or year, and filter searches during retrieval to ensure higher precision.

### 2. Retrieval Enhancements
- **Hybrid Search:** Combine the current dense vector search (embeddings) with keyword-based sparse search (BM25) to improve recall for exact keyword matching (e.g., retrieving specific policy IDs).
- **Semantic Chunking:** Upgrade from raw recursive text splitting to semantic chunking, which groups text by logical topic and meaning rather than arbitrary character counts.
- **Reranking Models:** Insert a cross-encoder model reranking step after retrieval to re-score and sort the top chunks for maximum relevance before passing them to the generator.

### 3. Application Architecture
- **API Decoupling:** Separate the LangChain backend into a RESTful or GraphQL API (e.g., using FastAPI) and keep Streamlit strictly as a frontend component. This allows other applications to query the HR RAG system programmatically.
- **Streaming Responses:** Implement native token streaming from the Ollama model to the Streamlit UI to vastly improve user experience and perceived wait times.
- **User Authentication:** Add login systems to manage employee-specific access levels, so sensitive executive policies are only queryable by authorized users.

## Author

**Ganesh Munigeti**

