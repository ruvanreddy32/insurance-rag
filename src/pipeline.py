from loader import load_documents,load_pdfs

from chunking import split_docs_by_recursive,split_docs_by_spacy,embed_chunks

data_urls = ["data/icici-health.pdf","data/lic-life.pdf"]

pages = load_pdfs(data_urls)
documents = load_documents(pages)
print("Loaded ",len(documents),"documents")
recursive_chunks = split_docs_by_recursive(documents)

spacy_chunks = split_docs_by_spacy(documents)


# print(len(recursive_chunks))
# print(len(spacy_chunks))

# print("recursive : ",recursive_chunks[0])

# print("spacy : ",spacy_chunks[0])
texts=[doc.page_content for doc in recursive_chunks]
embeddings=embed_chunks(texts)
print(embeddings.shape)
