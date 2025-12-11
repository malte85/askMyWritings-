import faiss
import numpy as np
import pickle
from openai import OpenAI
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="Ask My Writings API",
    description="RAG-enhanced LLM API for querying documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load OpenAI (will be initialized on startup if API key is available)
client = None

INDEX_PATH = "faiss_index.bin"
MAPPING_PATH = "faiss_mapping.pkl"

# Global variables for index and mapping
index = None
mapping = None
ids = None
docs = None


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
    retrieved_texts = [doc["content"] for doc in retrieved_docs]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer faithfully using the retrieved documents."},
            {"role": "user", "content": f"Question: {question}\n\nDocs:\n{retrieved_texts}"}
        ]
    )
    
    answer = response.choices[0].message.content
    
    return {
        "question": question,
        "answer": answer,
        "retrieved_docs": retrieved_docs
    }


@app.on_event("startup")
async def startup_event():
    """Load FAISS index and initialize OpenAI client on startup"""
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
