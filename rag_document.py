from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.llms import CTransformers
import requests

# Download PDF
url = "https://www.medicare.gov/publications/11514-A-Quick-Look-at-Medicare.pdf"
response = requests.get(url)
with open("medicare.pdf", "wb") as f:
    f.write(response.content)

# Load with PyPDFLoader
loader = PyPDFLoader("medicare.pdf")
docs = loader.load()

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)


llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    max_new_tokens=200
)
qa_model = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=-1, max_new_tokens=200)
# llm = HuggingFacePipeline(pipeline=qa_model)

# Build RetrievalQA
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
# Query
query = "What are the important deadlines for Medicare enrollment?"
answer = qa_chain.run(query)
print("Answer:", answer)

