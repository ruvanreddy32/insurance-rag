from langchain_text_splitters import RecursiveCharacterTextSplitter , SpacyTextSplitter
from typing import List
from langchain_core.documents import Document




def split_docs_by_recursive(docs:List[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return chunks


def split_docs_by_spacy(docs:List[Document]):
    splitter = SpacyTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        pipeline="en_core_web_sm"
    )
    chunks=splitter.split_documents(docs)
    return chunks


