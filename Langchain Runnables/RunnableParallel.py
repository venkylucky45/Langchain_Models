from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-120b')

template1 = PromptTemplate(
    template = 'Genarate a single but shhort tweet for {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Generate a single but short linkedin post for {topic} (dont give multiple options to choose from)',
    input_variables=['topic'] 
)

parser = StrOutputParser()

parallel = RunnableParallel({
    'tweet' : RunnableSequence(template1,model,parser),
    'post' : RunnableSequence(template2,model,parser)
})

result = parallel.invoke({'topic' : 'new job as AI engineer'})
print(result)