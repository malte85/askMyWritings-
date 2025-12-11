#!/usr/bin/env python3
"""
Example script demonstrating how to use the Ask My Writings API
"""
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_root():
    """Test the root endpoint"""
    print("Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_health():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_query(question, top_k=3, model="gpt-4o-mini"):
    """Test the query endpoint"""
    print(f"Querying: '{question}'...")
    
    payload = {
        "question": question,
        "top_k": top_k,
        "model": model
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Question: {data['question']}")
        print(f"Answer: {data['answer']}")
        print(f"Retrieved {len(data['retrieved_docs'])} documents")
        print()
    else:
        print(f"Error: {response.json()}\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Ask My Writings API - Example Client")
    print("=" * 60)
    print()
    
    # Test basic endpoints
    test_root()
    test_health()
    
    # Test query endpoint (will fail if index not loaded)
    print("Testing query endpoint...")
    test_query("What is the main topic?")
    
    print("=" * 60)
    print("Example complete!")
    print("=" * 60)
