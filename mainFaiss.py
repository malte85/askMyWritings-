import faiss
import numpy as np
import pickle
from openai import OpenAI
import os


# ✅ Load OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_PATH = "faiss_index.bin"
MAPPING_PATH = "faiss_mapping.pkl"

# Load FAISS index + mapping
index = faiss.read_index(INDEX_PATH)
with open(MAPPING_PATH, "rb") as f:
    mapping = pickle.load(f)

ids = mapping["ids"]
docs = mapping["docs"]

# Embed function
def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

while True:
    query = input("\nAsk a question (or 'quit'): ")
    if query.lower() in ["quit", "exit"]:
        break

    # Get embedding for query
    q_emb = embed_text(query)

    # Search FAISS
    D, I = index.search(q_emb, 3)  # top 3 results
    retrieved_docs = [docs[i] for i in I[0]]

    # Use OpenAI to answer based on retrieved docs
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer faithfully using the retrieved documents."},
            {"role": "user", "content": f"Question: {query}\n\nDocs:\n{retrieved_docs}"}
        ]
    )

    print("\nBOT:", response.choices[0].message.content)
