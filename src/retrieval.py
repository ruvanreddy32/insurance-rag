from sentence_transformers import SentenceTransformer

import chromadb

INSURER_KEYWORDS = {
    "lic": "lic",
    "jeevan": "lic",
    "icici": "icici",
    "lombard": "icici",
    "star health": "star",
    "hdfc": "hdfc",
    "ergo": "hdfc"
}

def detect_insurer(query):
    query_lower = query.lower()
    for keyword, insurer in INSURER_KEYWORDS.items():
        if keyword in query_lower:
            return insurer
    return None

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="data/vectordb")

collection=client.get_collection("insurance_docs")

def retrieve_chunks(query,k=5):

    insurer = detect_insurer(query)
    query_embeddings=model.encode([query])
    results=[]
    if insurer:
        results=collection.query(
        query_embeddings=query_embeddings,
        n_results=k,
        where={"insurer":insurer}

    )
    else:
        results=collection.query(
        query_embeddings=query_embeddings,
        n_results=k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    retrieved = []

    for doc, meta in zip(docs, metas):
        retrieved.append({
            "text": doc,
            "page": meta["page_num"],
            "source": meta["source"],
            "insurer":meta['insurer']
        })

    return retrieved