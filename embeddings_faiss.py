# embeddings_faiss.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import numpy as np
import faiss
import os
from typing import List

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def chunk_text(text: str, chunk_size=500, overlap=50):
    tokens = text.split()
    i = 0
    out = []
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        out.append(" ".join(chunk))
        i += chunk_size - overlap
    return out

def load_processed_jsonl(path: str) -> List[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text","")
            chunks = chunk_text(text)
            for idx, c in enumerate(chunks):
                metadata = dict(item)
                metadata.update({"chunk_id": idx})
                docs.append(Document(page_content=c, metadata=metadata))
    return docs

def build_faiss(processed_jsonl: str, out_dir: str="faiss_index"):
    model = SentenceTransformer(MODEL_NAME)
    docs = load_processed_jsonl(processed_jsonl)
    texts = [d.page_content for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # Build faiss index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    # Save metadata and texts
    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"meta": d.metadata, "text": d.page_content}, ensure_ascii=False) + "\n")
    print("Saved FAISS index to", out_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="processed.jsonl")
    p.add_argument("--out", default="faiss_index")
    args = p.parse_args()
    build_faiss(args.processed, args.out)
