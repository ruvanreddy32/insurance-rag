from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

def load_pdfs(data_urls):
    pages=[]
    for urls in data_urls:
        loader = PyMuPDFLoader(urls)
        reader= loader.load()

        for page in reader:
            text = page.page_content
            if text:
                pages.append({
                    "text" : text,
                    "page_num":page.metadata['page']+1,
                    "document":urls
                })
    return pages

def load_documents(pages):
    docs=[]
    for page in pages:
        doc = Document(
            page_content=page["text"],
            metadata={
                "source":page['document'],
                "page_num":page["page_num"]
            }

        )
        docs.append(doc)
    return docs



