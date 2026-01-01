from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-120b')

template1 = PromptTemplate(
    template='Give me example about {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template = 'Give me 5 important notes from {resp}',
    input_variables=['resp']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
res = chain.invoke({'topic' : 'AI'})
print(res)

chain.get_graph().print_ascii()