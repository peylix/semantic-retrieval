# Semantic Retrieval for Scientific Documents

This project investigates how modern neural embedding models and training strategies can significantly improve semantic retrieval performance for scientific documents. Starting from a traditional Word2Vec baseline, we progressively enhance retrieval accuracy through pre-trained sentence embeddings, supervised fine-tuning, and hard negative mining.

## Motivation

Traditional information retrieval methods such as TF-IDF or Word2Vec rely on shallow, context-independent representations. While computationally efficient, these methods struggle to capture the complex semantic relationships present in scientific text.

The goal of this project is to answer the following questions:

- How much improvement can modern sentence embedding models achieve over traditional Word2Vec retrieval?
- How effective is supervised fine-tuning for semantic retrieval tasks?
- Does incorporating hard negative samples further improve retrieval quality, and if so, how?

To answer these questions, we build and evaluate a semantic retrieval pipeline on the **SciFact** dataset using a series of increasingly powerful models.

## Dataset and Models

- **SciFact (BEIR – Generated Queries)**
  - A scientific fact verification dataset containing claims and supporting documents, augmented with automatically generated queries for retrieval evaluation.
  - Dataset link: https://huggingface.co/datasets/BeIR/scifact-generated-queries
  - The dataset is converted into **BEIR-compatible format** to enable standardized and reproducible retrieval evaluation.

- **Word2Vec**
  - A traditional word embedding model trained on the SciFact corpus.
  - Serves as a baseline for comparison.
  - Model link: https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html

- **all-MiniLM-L6-v2**
  - A pre-trained sentence embedding model from Hugging Face, designed for efficient semantic search.
  - Model link: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## How to Run

### Baseline Models

Before running the semantic retrieval notebook, you need to run the following baseline models to generate their results for comparison. Note that the `uv` package manager is used to handle dependencies and execution.

1. **Word2Vec**:
   ```bash
   uv run src/embedding_model/word2vec.py
   ```
   - Output: `results/word2vec_results.json`
2. **Zero-shot MiniLM**:
   ```bash
   uv run src/embedding_model/zero_shot.py
   ```
   - Output: `results/zero_shot_results.json`

### Jupyter Notebook

This project now provides a Jupyter Notebook version for easy demonstration and training on platforms like Google Colab.

We recommand using [Google Colab](https://colab.research.google.com/) for easy access to free GPU resources.

For running the notebook locally:

```bash
   uv run jupyter notebook semantic_retrieval.ipynb
   ```

## Models and Training Strategy

We evaluate four retrieval models with increasing levels of sophistication:

### 1. Word2Vec Baseline

- Static word embeddings trained on the corpus
- Sentence representations obtained via mean pooling
- Serves as a traditional baseline for comparison

### 2. Zero-shot MiniLM

- Pre-trained sentence embedding model (`all-MiniLM-L6-v2`)
- Used without any task-specific fine-tuning
- Demonstrates the strength of modern contextualized embeddings

### 3. Fine-tuned MiniLM (Standard)

- Based on the same MiniLM architecture
- Fine-tuned on the SciFact training data
- Trained using **MultipleNegativesRankingLoss**
- Learns task-specific semantic representations for retrieval

### 4. Fine-tuned MiniLM with Hard Negatives

- Extends standard fine-tuning by incorporating **hard negative samples**
- Hard negatives are semantically similar but non-relevant documents
- Encourages the model to learn finer-grained semantic distinctions

## Evaluation Metrics

All models are evaluated using standard BEIR retrieval metrics:

- Recall@10 / Recall@100
- Precision@10 / Precision@100
- MRR
- MAP@10 / MAP@100
- NDCG@10 / NDCG@100

## Results: Model Comparison

The following figure compares the retrieval performance of all four models across key metrics.

![All Models Comparison](figures/all_models_comparison.png)

### Key Observations

- **Word2Vec performs significantly worse** across all metrics, highlighting the limitations of static, context-independent embeddings.
- **Zero-shot MiniLM already achieves a large performance gain**, demonstrating the effectiveness of pre-trained sentence embeddings.
- **Fine-tuning further improves retrieval accuracy**, especially for ranking-based metrics such as MRR and NDCG.
- **Hard negative mining consistently provides additional improvements**, even though the absolute gains are modest.

---

## Impact of Hard Negative Mining

Hard negative mining improves retrieval performance by introducing more challenging negative examples during training. These negatives are closer to the decision boundary, forcing the model to learn more precise semantic distinctions rather than relying on easy negatives.

![Hard Negatives Impact](figures/hard_negatives_comparison.png)

Although the performance gains are relatively small, they are **consistent across all metrics**, which is expected when improving an already strong retrieval model.

## Key Takeaways

- Modern sentence embedding models dramatically outperform traditional Word2Vec retrieval
- Supervised fine-tuning is essential for adapting embeddings to domain-specific retrieval tasks
- Hard negative mining provides stable and meaningful improvements in high-performance regimes
- Training strategy and data quality are as important as model architecture in neural retrieval systems


## Limitations and Future Work

- All neural models adopt a bi-encoder architecture, limiting fine-grained token interactions
- The effectiveness of hard negatives depends on the quality of negative sample selection
- Future work may include:
  - Cross-encoder re-ranking for improved early precision
  - Adaptive or curriculum-based hard negative mining
  - Scaling to larger embedding models and domain-specific datasets

