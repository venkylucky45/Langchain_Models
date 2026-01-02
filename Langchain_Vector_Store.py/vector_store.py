from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

import os
# api=os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')

pc = Pinecone()

from pinecone import ServerlessSpec


if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
emb = OllamaEmbeddings(model = 'nomic-embed-text')

vector_store = PineconeVectorStore(index=index, embedding=emb)

loader = PyPDFLoader(r'C:\Users\VENKATESH\OneDrive\Desktop\Langchain Models\LangChain_DocLoader.py\previewFormOrdinaryDetail.pdf')
doc = loader.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_documents(doc)

# n=1
# for i in texts:
#     i.metadata['doc_no']=f'doc{n}'
#     n+=1
index.delete(delete_all=True)
vector_store.add_documents(texts)



results = vector_store.similarity_search(
    "What is Aadhaar number",
    k=5
)
for res in results:
    print('*'*40)
    print(f"* {res.page_content}")
# print(results.page_content)
