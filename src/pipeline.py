from loader import load_documents,load_pdfs

from chunking import split_docs_by_recursive,split_docs_by_spacy
from vector_store import embed_chunks,store_in_chroma

from retrieval import retrieve_chunks
from rag import ask

data_urls = ["data/icici-health.pdf","data/lic-life.pdf"]

pages = load_pdfs(data_urls)
documents = load_documents(pages)
recursive_chunks = split_docs_by_recursive(documents)

#spacy_chunks = split_docs_by_spacy(documents)

texts=[doc.page_content for doc in recursive_chunks]
embeddings,metadatas=embed_chunks(recursive_chunks)
store_in_chroma(texts,embeddings,metadatas)

query="what is the waiting period in icici"
retrived_chunks = retrieve_chunks(query)

response = ask(query,retrived_chunks)
print(response)
