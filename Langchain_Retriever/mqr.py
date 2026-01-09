# Multi Query Retriever
import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()
pc = Pinecone()
embeddings = OllamaEmbeddings(model='nomic-embed-text:latest')
index_name = os.getenv('INDEX_NAME')

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws',region='us-east-1')
    )

index = pc.Index(index_name)
print('Using Pinecone index:',index_name)

vector_store = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding = embeddings
)

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1", 'doc-id': 'doc-1'}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2", 'doc-id': 'doc-2'}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3", 'doc-id': 'doc-3'}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4", 'doc-id': 'doc-4'}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5", 'doc-id': 'doc-5'}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1", 'doc-id': 'doc-6'}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2", 'doc-id': 'doc-7'}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3", 'doc-id': 'doc-8'}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4", 'doc-id': 'doc-9'}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5", 'doc-id': 'doc-10'}),
]

vector_store.add_documents(all_docs)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={'k':5}),
    llm = ChatGroq(model='openai/gpt-oss-120b')
)

query = 'How to improve energy levels and maintain balance?'

similarity_retriever = vector_store.as_retriever(search_type='similarity',search_kwargss={'k':5})
similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)
multiquery_results = multiquery_results

for i, doc in enumerate(similarity_results):
    print(f'\n---Result {i+1}---')
    print(doc.page_content)
print('*'*80)

for i, doc in enumerate(multiquery_results):
    print(f'\n---MQR_Result {i+1} ---')
    print(doc.page_content)
index.delete(delete_all = True)