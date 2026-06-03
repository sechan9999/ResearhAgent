# Plan: LangChain RAG Learning

**Feature**: `langchain-rag-learning`  
**Phase**: Plan  
**Created**: 2026-06-02  
**Level**: Dynamic

---

## 1. Goal

Build working knowledge of the LangChain/LangGraph ecosystem through hands-on tutorials and documentation study, with a focus on RAG (Retrieval-Augmented Generation) pipelines, evaluation, orchestration, and observability.

---

## 2. Scope

| Topic | Source | Priority |
|-------|--------|----------|
| RAG pipeline (retrieval, chunking, embedding, generation) | LangChain RAG tutorial | High |
| RAG evaluation metrics (faithfulness, relevancy, context recall) | RAGAS docs | High |
| Stateful multi-step agent graphs | LangGraph quickstart | High |
| Tracing, debugging, evaluation in production | LangSmith docs | Medium |

---

## 3. Learning Objectives

1. **LangChain RAG Tutorial** — Build a complete RAG pipeline: document loading → chunking → vector store → retrieval chain → QA interface
2. **RAGAS** — Evaluate the RAG pipeline using automated metrics (no labeled data needed)
3. **LangGraph Quickstart** — Understand graph-based agent orchestration; extend the existing `research_agent.py`
4. **LangSmith** — Enable tracing on the local project; inspect runs, compare prompts, track latency

---

## 4. Deliverables

- [ ] `rag_pipeline.py` — end-to-end RAG with a local document corpus
- [ ] `rag_eval.py` — RAGAS evaluation script against the pipeline
- [ ] Extended `research_agent.py` with LangGraph improvements (checkpointing, better routing)
- [ ] LangSmith tracing enabled and verified (`.env` updated)
- [ ] `docs/02-design/features/langchain-rag-learning.design.md`

---

## 5. Dependencies

- Existing: `langchain`, `langgraph`, `langchain-openai`, `tavily-python`, `python-dotenv`
- New to install: `langchain-community`, `faiss-cpu` or `chromadb`, `ragas`, `langsmith`
- API keys: `OPENAI_API_KEY` (have), `TAVILY_API_KEY` (have), `LANGCHAIN_API_KEY` (need for LangSmith)

---

## 6. Out of Scope

- Production deployment
- Fine-tuning or custom embeddings
- Multi-modal RAG

---

## 7. Estimated Effort

| Phase | Est. Time |
|-------|-----------|
| RAG pipeline | 2–3 hrs |
| RAGAS evaluation | 1–2 hrs |
| LangGraph extension | 1–2 hrs |
| LangSmith setup | 30 min |
| **Total** | **~6–8 hrs** |
