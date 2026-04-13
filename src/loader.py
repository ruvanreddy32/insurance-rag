from pypdf import PdfReader


def load_pdfs(data_urls):
    documents=[]
    for urls in data_urls:
        reader = PdfReader(urls)
        print(reader.pages[0].extract_text())

        for page_num,page in enumerate(reader.pages):
            text = page.extract_text()

            if text:
                documents.append({
                    "text" : text,
                    "page_num":page_num,
                    "document":urls
                })
    return documents


data_urls = ["data/icici-health.pdf","data/lic-life.pdf"]

documents = load_pdfs(data_urls)
print("Loaded ",len(documents),"documents")

