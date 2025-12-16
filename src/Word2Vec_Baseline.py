from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Set, Any
import re
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# BEIR
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch


# =========================
# Part A: Tokenize
# =========================
_TOKEN_RE = re.compile(r"[^a-z0-9\s]+")

def tokenize(text: str) -> List[str]:
    text = str(text).lower().strip()
    text = _TOKEN_RE.sub("", text)
    return text.split()


# =========================
# Part B: Sentence embedding
# =========================
def sentence_embedding(text: str, w2v: Word2Vec) -> np.ndarray:
    tokens = tokenize(text)
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    if not vecs:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


# =========================
# Word2Vec Retriever (train + precompute corpus embeddings)
# =========================
class Word2VecRetriever:
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
            return self.documents  # fallback

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

                did = obj.get("_id", obj.get("id"))
                if did is None:
                    raise ValueError("corpus.jsonl line missing '_id' (or 'id').")

                title = obj.get("title", "")
                text = obj.get("text", "")
                content = (str(title).strip() + "\n" + str(text).strip()).strip()

                doc_ids.append(str(did))
                docs.append(content)

        return doc_ids, docs


# =========================
# BEIR wrapper for Word2Vec (encode_corpus / encode_queries)
# =========================
class Word2VecBEIRWrapper:
    def __init__(self, w2v_model: Word2Vec):
        self.w2v = w2v_model

    def _to_text(self, item: Any) -> str:
        if isinstance(item, dict):
            title = item.get("title", "")
            text = item.get("text", "")
            return (str(title).strip() + "\n" + str(text).strip()).strip()
        return str(item)

    def encode_corpus(self, corpus, batch_size: int = 64, **kwargs):
        texts = [self._to_text(doc) for doc in corpus]
        return np.vstack([sentence_embedding(t, self.w2v) for t in texts]).astype(np.float32)

    def encode_queries(self, queries, batch_size: int = 64, **kwargs):
        texts = [self._to_text(q) for q in queries]
        return np.vstack([sentence_embedding(t, self.w2v) for t in texts]).astype(np.float32)


# =========================
# Evaluate Word2Vec using BEIR (same metric set as zeroshot_metrics)
# =========================
def evaluate_word2vec_beir(
    beir_model: Any,
    data_dir: Path,
    split: str = "test",
    k: int = 10,
    batch_size: int = 64,
) -> Dict[str, float]:
    corpus, queries, qrels = GenericDataLoader(str(data_dir)).load(split=split)

    dres = DenseRetrievalExactSearch(beir_model, batch_size=batch_size)
    retriever = EvaluateRetrieval(dres, score_function="cos_sim")

    results = retriever.retrieve(corpus, queries)

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=[10, 100])

    def hit_and_mrr_at_k(k_val: int):
        hits, mrrs = [], []
        for qid, doc_scores in results.items():
            top_docs = [
                doc_id
                for doc_id, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k_val]
            ]
            rels = qrels.get(qid, {})
            rel_set = set(rels.keys()) if isinstance(rels, dict) else set(rels)

            hits.append(1.0 if any(d in rel_set for d in top_docs) else 0.0)

            mrr_val = 0.0
            for rank, doc_id in enumerate(top_docs, start=1):
                if doc_id in rel_set:
                    mrr_val = 1.0 / rank
                    break
            mrrs.append(mrr_val)

        n = len(hits) if hits else 1
        return sum(hits) / n, sum(mrrs) / n

    hit10, mrr10 = hit_and_mrr_at_k(10)

    metrics = {
        "NDCG@10": float(ndcg.get("NDCG@10", 0.0)),
        "NDCG@100": float(ndcg.get("NDCG@100", 0.0)),
        "MAP@10": float(_map.get("MAP@10", 0.0)),
        "MAP@100": float(_map.get("MAP@100", 0.0)),
        "Recall@10": float(recall.get("Recall@10", 0.0)),
        "Recall@100": float(recall.get("Recall@100", 0.0)),
        "Precision@10": float(precision.get("P@10", 0.0)),
        "Precision@100": float(precision.get("P@100", 0.0)),
        "Hit@10": float(hit10),
        "MRR": float(mrr10),
        "N": float(len(queries)),
    }
    return metrics


# =========================
# MAIN: run evaluation + save JSON in loader-compatible format
# =========================
if __name__ == "__main__":
    # ---- EDIT THIS to your local BEIR dataset root ----
    BEIR_DIR = Path("../data/processed/beir_format")  # must contain corpus.jsonl, queries.jsonl, qrels/train.tsv, qrels/test.tsv

    corpus_path = BEIR_DIR / "corpus.jsonl"
    train_qrels_path = BEIR_DIR / "qrels" / "train.tsv"

    # 1) Train Word2Vec on TRAIN relevant docs
    retriever = Word2VecRetriever(
        corpus_path=corpus_path,
        train_qrels_tsv_path=train_qrels_path,
        vector_size=300,
        window=5,
        min_count=2,
        sg=1,
        workers=4,
        model_path=None,      # optionally: Path("./results/word2vec.model")
        rebuild_model=False,
    )

    # 2) BEIR wrapper
    w2v_beir_model = Word2VecBEIRWrapper(retriever.w2v)

    # 3) Evaluate on TEST (k=10 + k=100 metrics)
    word2vec_metrics = evaluate_word2vec_beir(
        beir_model=w2v_beir_model,
        data_dir=BEIR_DIR,
        split="test",
        k=10,
        batch_size=64,
    )

    print("Word2Vec metrics:")
    print(word2vec_metrics)

    # 4) Save JSON so load_baseline_results() can read it
    # load_baseline_results() expects: {"model_name": ..., "metrics": {...}, ...}
    results_dir = Path("./word2vec_results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "word2vec_results.json"

    payload = {
        "model_name": "Word2Vec Baseline",
        "model_type": "Word2Vec",
        "base_model": "gensim-word2vec",
        "metrics": word2vec_metrics,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved loader-compatible JSON to:", out_path.resolve())