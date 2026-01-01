# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# load_dotenv()
# model = ChatGroq(
#     model = 'openai/gpt-oss-120b'
# )
# prompt = PromptTemplate(
#     template='Write a report about {topic} in {num} lines',
#     input_variables=['topic', 'num']
# )
# res=prompt.invoke({'topic':'LLM', 'num': 2})
# print(res)

# # chain = prompt | model

# result = model.invoke(res.text)
# print(result.content)
