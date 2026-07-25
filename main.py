from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


data = TextLoader("docs_loader/notes.txt")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system", "you are a ai that summarizes the text"), ("human", "{data}")]
)

model = ChatMistralAI(model="mistral-small-2506")
prompt = template.format_messages(data=docs[0].page_content)

result = model.invoke(prompt)
print(result.content)
