from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
embedding_model = MistralAIEmbeddings(model="mistral-embed")
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
)
llm = ChatMistralAI(model="mistral-small-2506")
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
                Use ONLY the provided context to answer the question.
                If the answer is not present in the context,
                say: "I could not find the answer in the document.""",
        ),
        ("human", "Context: {context} Question: {question}"),
    ]
)

# --- everything above this line is unchanged from your script ---
# --- delete the "while True" input loop below it, replace with this ---

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(q: Question):
    docs = retriever.invoke(q.question)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = template.invoke({"context": context, "question": q.question})
    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "sources": [
            {"content": d.page_content, "page": d.metadata.get("page")} for d in docs
        ],
    }
