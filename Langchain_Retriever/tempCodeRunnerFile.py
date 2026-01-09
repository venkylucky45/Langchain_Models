# mmr = Maximal Marginal Relevance
# Results relevant to the query while being diverse among themseleves
import os
from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
pc = Pinecone()
embeddings = OllamaEmbeddings(model='nomic-embed-text:latest')
index_name = os.getenv('INDEX_NAME')

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimensions=768,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws',region='us-east-1'),
    )
index = pc.Index(index_name)
print('Using Pinecone index:',index_name)

vector_store = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding = embeddings
)
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

for i, doc in enumerate(docs):
    doc.metadeta['doc_id'] = f'doc-{i+1}'

vector_store.add_documents(docs)

retriever_mmr = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k':3,'lambda_mult':0.6}
)

query = 'What is Langchain?'
results = retriever_mmr.invoke(query)

print('\n Vector Store MMr Retriever Results --->>')

for i, doc in enumerate(results):
    print(f'\n--Result {i+1}---')
    print(doc.page_content)
index.delete(delete_all=True)