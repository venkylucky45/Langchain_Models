from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-120b')

prompt = PromptTemplate(
    template = 'Write a 5 line summary for a poem \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader(r'C:\Users\VENKATESH\OneDrive\Desktop\Langchain Models\LangChain_DocLoader.py\cricket.txt',encoding='utf-8')

docs = loader.load()
# print(docs)
print(type(docs))
print(docs[0])
print(type(docs[0]))

chain = prompt | model | parser
res = chain.invoke({'poem' : docs[0].page_content})
print(res)
