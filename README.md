# Local Redis Semantic Cache for RAG

A Retrieval-Augmented Generation (RAG) pipeline that uses **Redis as a vector store** for document retrieval **and** as a **semantic cache** for previously answered questions — reducing latency and LLM calls for repeated or similar queries.

## Overview

This notebook builds an end-to-end RAG system over a PDF document, with a semantic caching layer sitting in front of the pipeline. When a user asks a question:

1. The system first checks a **Redis-backed semantic cache** for a similar question that was answered before.
2. If a close enough match is found (based on vector similarity score), the cached answer is returned instantly — skipping retrieval, reranking, and LLM generation entirely.
3. If no close match exists (cache miss), the full RAG pipeline runs: query rewriting → vector retrieval → cross-encoder reranking → context assembly → LLM generation. The new question/answer pair is then stored in the cache for future reuse.

## Pipeline Architecture

```
User Question
      │
      ▼
┌─────────────────────┐
│  Semantic Cache      │──── Cache Hit (score < threshold) ──► Return cached answer
│  (Redis Vector Store)│
└─────────────────────┘
      │ Cache Miss
      ▼
┌─────────────────────┐
│ Query Rewriting      │  (rewrites long/complex questions for better semantic search)
│ (Mistral LLM)         │
└─────────────────────┘
      ▼
┌─────────────────────┐
│ Vector Retrieval      │  (top-k similarity search over PDF chunks in Redis)
└─────────────────────┘
      ▼
┌─────────────────────┐
│ Cross-Encoder Rerank  │  (BAAI/bge-reranker-base reorders retrieved chunks)
└─────────────────────┘
      ▼
┌─────────────────────┐
│ Context Assembly      │  (formats top reranked chunks with source/page metadata)
└─────────────────────┘
      ▼
┌─────────────────────┐
│ LLM Answer Generation │  (Mistral `mistral-small-latest`)
└─────────────────────┘
      ▼
Store (question, answer) in semantic cache → Return answer
```

## Tech Stack

| Component | Tool / Library |
|---|---|
| Document loading | `langchain_community` (`PyPDFLoader`) |
| Chunking | `langchain_experimental` `SemanticChunker` (percentile breakpoint strategy) |
| Embeddings | `langchain_mistralai` (`mistral-embed`) |
| Vector store | Redis via `langchain-redis` (`RedisVectorStore`) |
| Semantic cache | Second Redis vector index (`rag_cach`), storing Q&A pairs |
| Reranking | `sentence_transformers` `CrossEncoder` (`BAAI/bge-reranker-base`) |
| LLM | `langchain_mistralai` `ChatMistralAI` (`mistral-small-latest`) |
| Cache backend | Redis (accessed locally via `redis-py`, tunneled through `ngrok`) |

## How It Works

### 1. Document Ingestion
A PDF (`/content/llm.pdf`) is loaded with `PyPDFLoader`, and irrelevant metadata fields (e.g. `producer`, `creationdate`, `subject`) are stripped from each page.

### 2. Semantic Chunking
Instead of fixed-size chunking, the notebook uses a `SemanticChunker` that splits text based on semantic similarity breakpoints (80th percentile threshold), producing more coherent, topic-consistent chunks.

### 3. Vector Storage (Redis)
Two separate Redis vector indexes are created:
- **`llm_paper`** — stores the semantic chunks of the source PDF for retrieval.
- **`rag_cach`** — stores previously asked questions (as vectors) mapped to their generated answers, acting as the semantic cache.

### 4. Semantic Cache Lookup (`ask()`)
For every incoming question, a similarity search (`k=1`) is run against the `rag_cach` index:
- If the closest match has a **distance score below `0.08`**, the cached answer is returned immediately (cache hit).
- Otherwise, the full `rag_pipeline()` is triggered (cache miss).

### 5. Full RAG Pipeline (`rag_pipeline()`)
On a cache miss:
- **Query rewriting**: questions longer than 12 words are rewritten by the LLM for better semantic search suitability.
- **Retrieval**: top-5 similar chunks are pulled from the `llm_paper` Redis index.
- **Reranking**: a cross-encoder (`BAAI/bge-reranker-base`) rescoring reorders the retrieved chunks by relevance to the (rewritten) query.
- **Context assembly**: the top reranked chunks are formatted with source and page metadata.
- **Generation**: the LLM answers strictly from the assembled context, refusing to answer if the information isn't present.
- **Caching**: the new (question, answer) pair is stored back into the semantic cache for future reuse.

The pipeline also prints a stage-by-stage timing breakdown (query transformation, retrieval, reranking, context creation, LLM generation, total time) for performance profiling.

## Setup

### Prerequisites
- Python 3.x (originally run on Google Colab with a T4 GPU, used for the cross-encoder)
- A running Redis instance (the notebook connects to a Redis server exposed via an `ngrok` TCP tunnel)
- A Mistral AI API key

### Installation
```bash
pip install langchain_community
pip install requests==2.32.4 --force-reinstall
pip install langchain_experimental
pip install pypdf
pip install langchain_mistralai
pip install sentence_transformers
pip install redis redisvl langchain-redis
```

### Environment Variables
Set the following as environment variables or Colab secrets rather than hardcoding them in the notebook:

```bash
MISTRAL_API_KEY=your_mistral_api_key
REDIS_HOST=your_redis_host
REDIS_PORT=your_redis_port
REDIS_PASSWORD=your_redis_password   # if required
```

> ⚠️ **Security note:** The original notebook has a Mistral API key and Redis connection details hardcoded directly in the cells. Before running or sharing this notebook, remove these hardcoded values and load them from environment variables or a secrets manager instead. If this key has already been shared/committed anywhere, rotate/revoke it immediately.

### Input Data
Place the source PDF at the path referenced in the loader cell (`/content/llm.pdf` for Colab), or update the path to point to your own document.

## Usage

```python
answer = ask("explain query, key, value")
print(answer)
```

- First call → cache miss → runs full pipeline, prints timing breakdown, stores result in cache.
- Repeated or semantically similar future calls → cache hit → instant cached answer, no LLM/retrieval cost.

## Configuration Notes

| Parameter | Location | Purpose |
|---|---|---|
| `breakpoint_threshold_amount=80` | `SemanticChunker` | Controls chunk granularity (higher = fewer, larger chunks) |
| `k=5` | `vectorstore.similarity_search` | Number of chunks retrieved before reranking |
| `k=1` | `semantic_cach.similarity_search_with_score` | Only the closest cached question is checked |
| `score < 0.08` | `ask()` | Similarity distance threshold for a cache hit — tune based on your embedding model's score distribution |
| `max_tokens=100` / `temperature=0.4` | `llm` (`ChatMistralAI`) | Controls answer length/creativity for the main generation call |

## Potential Improvements

- Externalize all secrets (API keys, Redis credentials) via `.env` / Colab secrets instead of hardcoding.
- Add a TTL (time-to-live) or eviction policy for cached Q&A entries in Redis to avoid stale or unbounded cache growth.
- Make the cache similarity threshold configurable/tested against a labeled set of paraphrased questions to tune precision vs. recall of cache hits.
- Add error handling around Redis connectivity (the current `ping()` check has no retry/fallback logic).
- Log cache hit/miss rates over time to quantify the actual latency and cost savings from semantic caching.

## License
Add your preferred license here (e.g. MIT).
