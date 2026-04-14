import chromadb
from sentence_transformers import SentenceTransformer

def embed_chunks(chunks):
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    for data in metadatas:
        data['insurer']=get_insurer_name(data['source'])
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings=embedding_model.encode(texts).tolist()
    return embeddings,metadatas

def get_insurer_name(filename):
    filename = filename.lower()
    if "lic" in filename: return "lic"
    if "icici" in filename: return "icici"
    if "star" in filename: return "star"
    if "hdfc" in filename: return "hdfc"
    return "unknown"

client = chromadb.PersistentClient(path="data/vectordb")

collection=client.get_or_create_collection("insurance_docs")

def store_in_chroma(texts,embeddings,metadatas):
    ids=[f"{data['source']}_chunk_{i}" for i, data in enumerate(metadatas)]
    print(metadatas[:5])
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print("succesfully stored in chrmoa")
