import os
from pathlib import Path
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Get Pinecone index name from environment
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME environment variable is required")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

pc = Pinecone()



script_dir = Path(__file__).parent
pdf_path = script_dir.parent / "sample_rag_nist_ai_rmf_1_0.pdf"

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

loader = PyPDFLoader(str(pdf_path))
print("loading document.....")

documents = loader.load()

print("document loaded.....")

print("splitting document.....")
chunks = text_splitter.split_documents(documents)
print("splitting document done.....")

texts = [chunk.page_content for chunk in chunks]
print("encoding document.....")
vectors = embedding_model.encode(texts, normalize_embeddings=True) 
print("encoding document done.....")

to_upsert = []
id = 0
for chunk, vector in zip(chunks, vectors):
    chunk_id = f"chunk_{id}"
    id += 1
    chunk_metadata = chunk.metadata
    chunk_page_content = chunk.page_content
    chunk_metadata["page_content"] = chunk_page_content
    to_upsert.append({
        "id": chunk_id,
        "values": vector.tolist(),
        "metadata": chunk_metadata
    })

print(f"Upserting {len(to_upsert)} chunks into Pinecone")

index = pc.Index(PINECONE_INDEX_NAME)

PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
if not PINECONE_NAMESPACE:
    raise ValueError("PINECONE_NAMESPACE environment variable is required")
# Batch size for upserting
BATCH_SIZE = 10
for i in range(0, len(to_upsert), BATCH_SIZE):
    print(f"Upserting batch {i//BATCH_SIZE+1} of {len(to_upsert)//BATCH_SIZE}")
    index.upsert(vectors=to_upsert[i:i+BATCH_SIZE], namespace=PINECONE_NAMESPACE)