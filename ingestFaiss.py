import os
import faiss
import pickle
from openai import OpenAI
from pypdf import PdfReader

# ✅ Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_PATH = "faiss_index.bin"
MAPPING_PATH = "faiss_mapping.pkl"

# Helper: chunk text
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Helper: read PDF
def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Embed function
def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

# Ingest documents
DOCS_PATH = "my_docs/"
documents = []
ids = []

for filename in os.listdir(DOCS_PATH):
    path = os.path.join(DOCS_PATH, filename)

    if filename.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif filename.endswith(".pdf"):
        text = read_pdf(path)
    else:
        continue

    chunks = chunk_text(text)
    documents.extend(chunks)
    ids.extend([f"{filename}_{i}" for i in range(len(chunks))])

# Compute embeddings
embeddings = embed_texts(documents)

# Build FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype="float32"))

# Save index and mapping
faiss.write_index(index, INDEX_PATH)
with open(MAPPING_PATH, "wb") as f:
    pickle.dump({"ids": ids, "docs": documents}, f)

print(f"✅ Ingested {len(documents)} chunks into FAISS index.")
