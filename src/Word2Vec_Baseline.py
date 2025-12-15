from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Set
import re
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Part A: Tokenize
_TOKEN_RE = re.compile(r"[^a-z0-9\s]+")

def tokenize(text: str) -> List[str]:
    text = str(text).lower().strip()
    text = _TOKEN_RE.sub("", text)
    return text.split()

# Part B: Sentence embedding
def sentence_embedding(text: str, w2v: Word2Vec) -> np.ndarray:
    tokens = tokenize(text)
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    if not vecs:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


class Word2VecRetriever:
    """
    Train Word2Vec on TRAIN qrels documents only (no test leakage),
    but build retrieval index on ALL corpus documents.

    corpus.jsonl format (BEIR-like):
      {"_id": "...", "title": "...", "text": "..."}  (or "id" instead of "_id")

    qrels/train.tsv format (NO header):
      query_id \t doc_id \t relevance
    """

    def __init__(
        self,
        corpus_path: Union[str, Path],
        train_qrels_tsv_path: Optional[Union[str, Path]] = None,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 2,
        sg: int = 1,
        workers: int = 4,
        model_path: Optional[Union[str, Path]] = None,
        rebuild_model: bool = False,
    ):
        self.corpus_path = Path(corpus_path)
        self.train_qrels_tsv_path = Path(train_qrels_tsv_path) if train_qrels_tsv_path else None
        self.model_path = Path(model_path) if model_path is not None else None

        # 1) Load full corpus (for retrieval)
        self.doc_ids, self.documents = self._load_corpus_jsonl(self.corpus_path)
        self.id2doc = dict(zip(self.doc_ids, self.documents))

        # 2) Build training texts from train qrels (train-only)
        train_texts = self._build_training_texts_from_train_qrels()

        # 3) Train/load Word2Vec
        if self.model_path is not None and self.model_path.exists() and not rebuild_model:
            self.w2v = Word2Vec.load(str(self.model_path))
        else:
            tokenized_train = [tokenize(t) for t in train_texts]
            self.w2v = Word2Vec(
                sentences=tokenized_train,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                sg=sg,
                workers=workers,
            )
            if self.model_path is not None:
                self.w2v.save(str(self.model_path))

        # 4) Precompute embeddings for ALL docs (index)
        self.corpus_embeddings = np.vstack(
            [sentence_embedding(doc, self.w2v) for doc in self.documents]
        ).astype(np.float32)

    def _build_training_texts_from_train_qrels(self) -> List[str]:
        if self.train_qrels_tsv_path is None:
            # fallback (not recommended): use full corpus
            return self.documents

        df = pd.read_csv(
            self.train_qrels_tsv_path,
            sep="\t",
            header=None,
            names=["query_id", "doc_id", "relevance"],
        )

        train_doc_ids: Set[str] = set(df.loc[df["relevance"] > 0, "doc_id"].astype(str).tolist())
        train_texts = [self.id2doc[d] for d in train_doc_ids if d in self.id2doc]

        if not train_texts:
            raise ValueError("No training docs matched corpus doc_ids. Check id formats.")
        return train_texts

    @staticmethod
    def _load_corpus_jsonl(corpus_path: Path) -> Tuple[List[str], List[str]]:
        doc_ids: List[str] = []
        docs: List[str] = []

        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)

                # BEIR usually uses "_id"
                did = obj.get("_id", obj.get("id"))
                if did is None:
                    raise ValueError("corpus.jsonl line missing '_id' (or 'id').")

                title = obj.get("title", "")
                text = obj.get("text", "")
                content = (str(title).strip() + "\n" + str(text).strip()).strip()

                doc_ids.append(str(did))
                docs.append(content)

        return doc_ids, docs

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        q_emb = sentence_embedding(query, self.w2v).reshape(1, -1)
        sims = cosine_similarity(q_emb, self.corpus_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.doc_ids[i] for i in top_idx]
