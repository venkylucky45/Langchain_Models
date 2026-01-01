from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

template = PromptTemplate(
    template = 'give simple joke on {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableSequence(template,model,parser)

result = chain.invoke({'topic':'Akbar Birbal'})
print(result)