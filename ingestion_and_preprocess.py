# ingestion_and_preprocess.py
import os
import re
import json
import pandas as pd
import pdfplumber
from sqlalchemy import create_engine
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

# Basic PII regexes (extend per requirements)
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # names can't be reliably masked, but we can optionally mask common title patterns:
    "mr_mrs": re.compile(r"\b(Mr|Ms|Mrs|Dr)\.?\s+[A-Z][a-z]+\b")
}

def mask_pii(text: str, replacement="<REDACTED>") -> str:
    t = text
    for name, pattern in PII_PATTERNS.items():
        t = pattern.sub(replacement, t)
    return t

def clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def load_pdf(path: str) -> List[Dict]:
    docs = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            txt = mask_pii(clean_text(txt))
            if txt.strip():
                docs.append({"source": path, "page": i+1, "text": txt})
    return docs

def load_txt(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = mask_pii(clean_text(f.read()))
    return [{"source": path, "page": 0, "text": txt}]

def load_csv(path: str, text_cols: List[str]=None) -> List[Dict]:
    df = pd.read_csv(path)
    if text_cols is None:
        # pick all string columns
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
    docs = []
    for idx, row in df.iterrows():
        combined = " ".join([str(row[c]) for c in text_cols if pd.notna(row[c])])
        combined = mask_pii(clean_text(combined))
        docs.append({"source": path, "row": int(idx), "text": combined})
    return docs

def load_json(path: str, text_key: str="text") -> List[Dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # obj might be dict or list
    if isinstance(obj, dict):
        obj = [obj]
    for i, item in enumerate(obj):
        txt = item.get(text_key) or " ".join(str(v) for v in item.values())
        txt = mask_pii(clean_text(txt))
        docs.append({"source": path, "index": i, "text": txt})
    return docs

def load_sql_table(conn_uri: str, table: str, text_cols: List[str]=None, limit: int=None) -> List[Dict]:
    engine = create_engine(conn_uri)
    with engine.connect() as conn:
        df = pd.read_sql_table(table, conn, columns=text_cols) if text_cols else pd.read_sql_table(table, conn)
    if limit:
        df = df.head(limit)
    return load_csv_from_df(df, source=f"sql:{table}")

def load_csv_from_df(df: pd.DataFrame, source="df") -> List[Dict]:
    docs = []
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    for idx, row in df.iterrows():
        combined = " ".join([str(row[c]) for c in text_cols if pd.notna(row[c])])
        combined = mask_pii(clean_text(combined))
        docs.append({"source": source, "row": int(idx), "text": combined})
    return docs

def ingest_folder(folder: str) -> List[Dict]:
    docs = []
    for root, _, files in os.walk(folder):
        for fname in files:
            path = os.path.join(root, fname)
            ext = Path(fname).suffix.lower()
            try:
                if ext == ".pdf":
                    docs.extend(load_pdf(path))
                elif ext in [".txt", ".text"]:
                    docs.extend(load_txt(path))
                elif ext == ".csv":
                    docs.extend(load_csv(path))
                elif ext == ".json":
                    docs.extend(load_json(path))
                else:
                    # skip unknown
                    continue
            except Exception as e:
                print(f"failed to ingest {path}: {e}")
    return docs

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="data", help="input folder with files")
    p.add_argument("--out", default="processed.jsonl")
    args = p.parse_args()
    docs = ingest_folder(args.folder)
    # write to jsonl
    with open(args.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} docs to {args.out}")
