from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Union, Optional
import re
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

# Part B: Sentence embedding (mean pooling over word vectors)
def sentence_embedding(text: str, w2v: Word2Vec) -> np.ndarray:
    tokens = tokenize(text)
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    if not vecs:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)

# Part C: Word2VecRetriever
# - 读取 corpus_csv (doc_id, content)
# - 训练或加载 Word2Vec
# - 预计算 corpus embeddings
# - retrieve(query) -> top-k doc_id list
class Word2VecRetriever:
    def __init__(
        self,
        corpus_csv_path: Union[str, Path],
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 2,
        sg: int = 1,
        workers: int = 4,
        model_path: Optional[Union[str, Path]] = None,
        rebuild_model: bool = False,
    ):
        """
        Parameters
        ----------
        corpus_csv_path:
            preprocess.py 生成的 scifact_corpus.csv（必须含 doc_id, content 两列）
        vector_size/window/min_count/sg/workers:
            Word2Vec 训练参数
        model_path:
            如果提供，则尝试从该路径 load Word2Vec，避免每次重新训练
        rebuild_model:
            如果 True，则即使 model_path 存在也强制重新训练
        """
        self.corpus_csv_path = Path(corpus_csv_path)
        self.model_path = Path(model_path) if model_path is not None else None

        # 1) Load corpus
        self.doc_ids, self.documents = self._load_corpus(self.corpus_csv_path)

        # 2) Train or load Word2Vec
        if self.model_path is not None and self.model_path.exists() and not rebuild_model:
            self.w2v = Word2Vec.load(str(self.model_path))
        else:
            tokenized_docs = [tokenize(doc) for doc in self.documents]
            self.w2v = Word2Vec(
                sentences=tokenized_docs,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                sg=sg,          # sg=1: skip-gram; sg=0: CBOW
                workers=workers,
            )
            if self.model_path is not None:
                self.w2v.save(str(self.model_path))

        # 3) Precompute embeddings for all docs (N x D)
        self.corpus_embeddings = np.vstack(
            [sentence_embedding(doc, self.w2v) for doc in self.documents]
        ).astype(np.float32)

    @staticmethod
    def _load_corpus(corpus_csv_path: Path) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(corpus_csv_path)
        # 需要两列：doc_id, content
        if "doc_id" not in df.columns or "content" not in df.columns:
            raise ValueError("corpus_csv must contain columns: 'doc_id' and 'content'")
        doc_ids = df["doc_id"].astype(str).tolist()
        documents = df["content"].astype(str).tolist()
        return doc_ids, documents

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        输入 query 文本，返回按相似度降序排列的 top_k doc_id
        """
        q_emb = sentence_embedding(query, self.w2v).reshape(1, -1)
        sims = cosine_similarity(q_emb, self.corpus_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.doc_ids[i] for i in top_idx]
    



  #retriever = Word2VecRetriever(
     # corpus_csv_path="data/processed/scifact_corpus.csv",
      #model_path="data/processed/w2v.model",   # 可选
  #)

  #top_docs = retriever.retrieve("Does aspirin reduce cancer risk?", top_k=10)
  #print(top_docs)


