# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

# model = ChatOllama(model = 'gemma3:4b')
model = ChatGroq(model = 'openai/gpt-oss-120b')
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction' : parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = 'Write a single appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write a single appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

branch_main = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: 'could not find sentiment')
)

chain = classifier_chain | branch_main

res = chain.invoke({'feedback' : 'I like you service'})
print(res)