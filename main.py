from dotenv import load_dotenv
form langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")

res = model.invoke("Hello!")
print(res.content)