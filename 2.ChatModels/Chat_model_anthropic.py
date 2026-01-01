from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
model = ChatAnthropic(model='openai/gpt-oss-120b')
result = model.invoke('What is the capital of India')
print(result.content)