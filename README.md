# askMyWritings-

API backend for RAG-enhanced document querying using FAISS and OpenAI.

## Overview

This repository provides a FastAPI-based REST API for querying documents using Retrieval-Augmented Generation (RAG). It uses FAISS for efficient vector similarity search and OpenAI's language models for generating responses.

## Features

- **FastAPI REST API**: Modern, fast API framework with automatic OpenAPI documentation
- **RAG System**: Retrieval-Augmented Generation using FAISS vector search
- **OpenAI Integration**: Uses OpenAI embeddings and chat completions
- **Document Ingestion**: Support for PDF and text files
- **Health Checks**: Built-in health and status endpoints

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/malte85/askMyWritings-.git
cd askMyWritings-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Prepare Documents

1. Create a `my_docs/` directory and add your documents (PDF or TXT files):
```bash
mkdir my_docs
# Add your .pdf or .txt files to my_docs/
```

2. Run the ingestion script to create the FAISS index:
```bash
python ingestFaiss.py
```

This will create `faiss_index.bin` and `faiss_mapping.pkl` files.

## Usage

### Running the API Server

Start the API server:
```bash
python api.py
```

Or using uvicorn directly:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`.

### API Endpoints

#### Root Endpoint
```bash
GET /
```
Returns API status and version.

**Example:**
```bash
curl http://localhost:8000/
```

#### Health Check
```bash
GET /health
```
Returns server health status and index information.

**Example:**
```bash
curl http://localhost:8000/health
```

#### Query Documents
```bash
POST /query
```
Query the RAG system with a question.

**Request Body:**
```json
{
  "question": "What is the main topic?",
  "top_k": 3,
  "model": "gpt-4o-mini"
}
```

**Parameters:**
- `question` (required): The question to ask
- `top_k` (optional): Number of documents to retrieve (default: 3)
- `model` (optional): OpenAI model to use (default: "gpt-4o-mini")

**Response:**
```json
{
  "question": "What is the main topic?",
  "answer": "Based on the retrieved documents...",
  "retrieved_docs": [
    {
      "content": "Document text...",
      "index": 0
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of my writings?"}'
```

### Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Legacy Command-Line Interface

The original command-line interface is still available in `mainFaiss.py`:

```bash
python mainFaiss.py
```

This provides an interactive question-answering loop in the terminal.

## Project Structure

- `api.py` - FastAPI application and endpoints
- `mainFaiss.py` - Original CLI interface for querying
- `ingestFaiss.py` - Document ingestion and FAISS index creation
- `requirements.txt` - Python dependencies
- `faiss_index.bin` - FAISS vector index (generated)
- `faiss_mapping.pkl` - Document mapping (generated)
- `my_docs/` - Directory for source documents (to be created)

## Development

### Running in Development Mode

For development with auto-reload:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `ALLOWED_ORIGINS` - Comma-separated list of allowed CORS origins (optional, defaults to "*" for all origins)

**Example:**
```bash
export OPENAI_API_KEY='your-api-key-here'
export ALLOWED_ORIGINS='http://localhost:3000,https://example.com'
```

## Error Handling

The API includes comprehensive error handling:

- **503 Service Unavailable**: FAISS index not loaded
- **500 Internal Server Error**: Error processing query
- **422 Validation Error**: Invalid request format

## License

This project is provided as-is for educational and research purposes.
 
