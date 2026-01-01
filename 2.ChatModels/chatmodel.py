from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b')

result = model.invoke('What is the capital of India?')
print(result.content)

# from dotenv import load_dotenv
# from langchain_groq import ChatGroq

# load_dotenv()

# model = ChatGroq(
#     model="mixtral-8x7b-32768"
# )

# result = model.invoke("What is the capital of India?")
# print(result)
