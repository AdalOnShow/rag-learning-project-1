from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("docs_loader/GRU.pdf")
docs = data.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=2)
chunks = splitter.split_documents(docs)

print(chunks[20].page_content)

# print(docs[14])
