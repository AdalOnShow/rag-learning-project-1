from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"},
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "DataScience_book"},
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_book"},
    ),
]

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="vector_store/chroma_db",
)
