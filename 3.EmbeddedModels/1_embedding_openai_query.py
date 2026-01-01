from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)

result = embedding.embed_query("delhi is the capital of India")

print(len(result))
print(result[:10])
