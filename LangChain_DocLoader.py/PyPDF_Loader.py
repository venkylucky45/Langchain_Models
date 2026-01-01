from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r'C:\Users\VENKATESH\OneDrive\Desktop\Langchain Models\LangChain_DocLoader.py\previewFormOrdinaryDetail.pdf')

docs = loader.load()
print(docs)