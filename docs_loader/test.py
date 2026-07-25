from langchain_community.document_loaders import TextLoader

data = TextLoader("docs_loader/notes.txt")
docs = data.load()

print(docs)
