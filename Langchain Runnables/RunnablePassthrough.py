from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnableParallel
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-120b')

template1 = PromptTemplate(
    template = 'Generate a small joke on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template = 'Explain this joke - {joke}',
    input_variables=['joke']
)

parser = StrOutputParser()

jokeGenerator = RunnableSequence(template1,model,parser)

meaningGen = RunnableParallel({
    'joke' : RunnableLambda(lambda x: x),
    'meaning' : RunnableSequence(template2,model,parser)
})

chain = RunnableSequence(jokeGenerator,meaningGen)

res = chain.invoke({'topic':'cricket'})
print(res)
