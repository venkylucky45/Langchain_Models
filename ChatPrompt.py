from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv
load_dotenv()
# model = ChatGroq(
#     model = 'openai/gpt-oss-120b'
#     temperature=0.5
# )

# prompt = ChatPromptTemplate([
#     ('system','You are a professional {domain} expert'),
#     ('human','Explain what is {topic}')
# ])

# promp = prompt.invoke({'domain':'AI','topic':'LLM'})
# print(promp)

li = [SystemMessage(content='You are a math expert'), HumanMessage(content='What is 2+2'), AIMessage(content = '2+2 is 4')]

print(li)