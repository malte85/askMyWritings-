import faiss
import numpy as np
import pickle
from openai import OpenAI
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional


# ✅ Load OpenAI (will be initialized on startup if API key is available)
client = None

INDEX_PATH = "faiss_index.bin"
MAPPING_PATH = "faiss_mapping.pkl"

# Global variables for index and mapping
index = None
mapping = None
ids = None
docs = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup: Load FAISS index and initialize OpenAI client
    global client, index, mapping, ids, docs
    
    try:
        initialize_client()
        print("✅ OpenAI client initialized successfully")
    except ValueError as e:
        print(f"⚠️  Warning: {e}")
    except Exception as e:
        print(f"❌ Error initializing OpenAI client: {e}")
    
    try:
        load_index()
        print("✅ FAISS index loaded successfully")
    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}")
    except Exception as e:
        print(f"❌ Error loading index: {e}")
    
    yield
    
    # Shutdown: cleanup if needed
    print("Shutting down...")


app = FastAPI(
    title="Ask My Writings API",
    description="RAG-enhanced LLM API for querying documents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
# NOTE: For production, replace ["*"] with specific allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),  # Configure via environment variable
    allow_credentials=False,  # Disabled when using wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request model for querying the RAG system"""
    question: str
    top_k: Optional[int] = 3
    model: Optional[str] = "gpt-4o-mini"


class RetrievedDocument(BaseModel):
    """Model for a retrieved document"""
    content: str
    index: int


class QueryResponse(BaseModel):
    """Response model for query results"""
    question: str
    answer: str
    retrieved_docs: List[RetrievedDocument]


def initialize_client():
    """Initialize OpenAI client"""
    global client
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)


def load_index():
    """Load FAISS index and mapping"""
    global index, mapping, ids, docs
    
    if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(
            f"FAISS index files not found. Please run ingestFaiss.py first to create {INDEX_PATH} and {MAPPING_PATH}"
        )
    
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, "rb") as f:
        mapping = pickle.load(f)
    
    ids = mapping["ids"]
    docs = mapping["docs"]


def embed_text(text: str) -> np.ndarray:
    """Generate embedding for a given text using OpenAI"""
    if client is None:
        raise RuntimeError("OpenAI client not initialized")
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)


def query_rag(question: str, top_k: int = 3, model: str = "gpt-4o-mini") -> dict:
    """
    Query the RAG system with a question
    
    Args:
        question: The question to ask
        top_k: Number of documents to retrieve (default: 3)
        model: OpenAI model to use (default: gpt-4o-mini)
    
    Returns:
        Dictionary containing the answer and retrieved documents
    """
    if index is None or docs is None:
        raise RuntimeError("Index not loaded. Server may not be initialized properly.")
    
    # Get embedding for query
    q_emb = embed_text(question)
    
    # Search FAISS
    D, I = index.search(q_emb, top_k)
    retrieved_docs = [{"content": docs[i], "index": int(i)} for i in I[0]]
    
    # Use OpenAI to answer based on retrieved docs
    formatted_docs = "\n\n---\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(retrieved_docs)])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer faithfully using the retrieved documents."},
            {"role": "user", "content": f"Question: {question}\n\nRetrieved Documents:\n\n{formatted_docs}"}
        ]
    )
    
    answer = response.choices[0].message.content
    
    return {
        "question": question,
        "answer": answer,
        "retrieved_docs": retrieved_docs
    }


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Ask My Writings API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    index_loaded = index is not None and docs is not None
    return {
        "status": "healthy" if index_loaded else "degraded",
        "index_loaded": index_loaded,
        "num_documents": len(docs) if docs else 0
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with a question
    
    Args:
        request: QueryRequest containing the question and optional parameters
    
    Returns:
        QueryResponse with the answer and retrieved documents
    """
    if index is None or docs is None:
        raise HTTPException(
            status_code=503,
            detail="FAISS index not loaded. Please ensure the index files exist and restart the server."
        )
    
    try:
        result = query_rag(
            question=request.question,
            top_k=request.top_k,
            model=request.model
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
