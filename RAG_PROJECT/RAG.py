from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
load_dotenv()
load = PyPDFLoader(r'C:\Users\VENKATESH\OneDrive\Desktop\Langchain Models\RAG_PROJECT\Rushikesh Kulkarni (3).pdf')
documents = load.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)
chunks = text_splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model = 'nomic-embed-text:latest')
vector_db = FAISS.from_documents(chunks,embeddings)
retriever = vector_db.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k':4}
)


model = ChatGroq(model='openai/gpt-oss-120b',temperature=0,max_tokens=300)
rag_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type = 'stuff',
    return_source_documents=True
)

query = 'Which experience does Rushikesh have in AWS and data engineering?'
response = rag_chain(query)
print('Answer')
print(response['result'])