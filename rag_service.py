# rag_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import pipeline
from typing import List, Dict
import json
import os

app = FastAPI()

class QueryIn(BaseModel):
    query: str
    top_k: int = 3

# load FAISS from your saved artifacts (example for LangChain-FAISS wrapper)
# NOTE: adapt path to where you saved the index via LangChain (if you used raw FAISS, implement mapping load)
FAISS_DIR = "faiss_langchain"  # if you used LangChain's FAISS.from_documents().save_local()
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load embeddings wrapper and vectorstore
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = FAISS.load_local(FAISS_DIR, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM: use CTransformers wrapper (fast) or a HF pipeline
from langchain_community.llms import CTransformers
llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    max_new_tokens=256
)

# Template: ask LLM to be concise and show sources
PROMPT = PromptTemplate(input_variables=["context", "question"],
                        template="""You are an expert medical assistant. Use ONLY the information in the context to answer.
If the answer is not contained in the context, say "I don't know — please consult clinical resources".
Context:
{context}

Question: {question}
Answer concisely and list which source (metadata) you used for each claim.
""")

chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True, chain_type_kwargs={"prompt":PROMPT})

@app.post("/query")
def query(q: QueryIn):
    try:
        result = chain({"query": q.query})
        answer = result["result"]
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({"text": doc.page_content[:300], "meta": doc.metadata})
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
