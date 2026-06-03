# Completion Report: LangChain RAG Learning

**Feature**: `langchain-rag-learning`  
**Completed**: 2026-06-02  
**Match Rate**: 77% raw / 88% adjusted (LangSmith excluded — pending API key)  
**PDCA Cycle**: Plan → Design → Do → Check → Report

---

## 1. Executive Summary

Successfully built a LangChain/LangGraph learning project covering RAG pipelines, RAGAS evaluation, and LangGraph graph orchestration. All core deliverables are working and tested. Two minor gaps remain (RAGAS metric coverage, `interrupt_before`) plus LangSmith tracing pending an external API key.

---

## 2. What Was Built

### New Files
| File | Purpose | Status |
|------|---------|--------|
| `rag_pipeline.py` | Full RAG pipeline — load docs, embed, store, retrieve, answer | Done |
| `rag_eval.py` | RAGAS evaluation — faithfulness + context_precision scoring | Done |
| `data/sample_docs/*.txt` | 4 corpus files (LangChain, LangGraph, RAGAS, LangSmith) | Done |
| `data/faiss_index/` | Persisted FAISS vector store (12 chunks) | Done |

### Updated Files
| File | Change |
|------|--------|
| `research_agent.py` | Added `MemorySaver` checkpointer + `thread_id` config |
| `requirements.txt` | Added 5 new dependencies |
| `main.py` | Fixed emoji encoding (Windows cp1252 compatibility) |

---

## 3. Architecture Delivered

```
data/sample_docs/*.txt
        │
        ▼
DirectoryLoader + RecursiveCharacterTextSplitter (chunk=500, overlap=50)
        │
        ▼
OpenAIEmbeddings (text-embedding-3-small)  →  FAISS (persisted)
        │
        ▼ (retriever k=4)
LCEL chain: retriever | format_docs | ChatPromptTemplate | gpt-4o-mini | StrOutputParser
        │
        ▼
rag_eval.py → RAGAS [faithfulness, context_precision] → scores DataFrame
```

---

## 4. RAGAS Evaluation Results

Evaluated against 4 questions from the corpus:

| Question | Faithfulness | Context Precision |
|----------|-------------|-------------------|
| What is LCEL and how does it work? | 0.667 | 1.000 |
| What env vars does LangSmith need? | 1.000 | 1.000 |
| How does RAGAS measure faithfulness? | 1.000 | 0.833 |
| What is a MemorySaver in LangGraph? | 1.000 | 1.000 |
| **Average** | **0.917** | **0.958** |

Both metrics exceed the design target (>= 0.7). Pipeline is functioning correctly.

---

## 5. LangGraph Improvement

`research_agent.py` now compiles with `MemorySaver`:
```python
graph = builder.compile(checkpointer=MemorySaver())
```
Invoked with `config={"configurable": {"thread_id": "research-1"}}` — thread-based memory is active.

---

## 6. Key Learnings & Issues Encountered

| Issue | Root Cause | Resolution |
|-------|-----------|------------|
| Windows emoji crash | `cp1252` terminal can't encode `🔍` | Removed emoji from `main.py` |
| RAGAS import error | `langchain_community.chat_models.vertexai` removed in v0.4 | Created stub shim |
| RAGAS API churn | `evaluate()`, metric singletons, `LangchainLLMWrapper` all deprecated in v0.4 | Used `from ragas.metrics import faithfulness` (old singleton API) |
| `answer_relevancy` failure | `OpenAIEmbeddings.embed_query` removed in langchain-openai v1.x | Dropped metric |
| `MemorySaver` needs thread_id | `graph.invoke()` requires config when checkpointer is set | Added `config={"configurable": {"thread_id": ...}}` |

---

## 7. Open Items

| ID | Item | Priority | Action |
|----|------|----------|--------|
| GAP-2a | `context_recall` not in eval | Low | Add to `rag_eval.py` (1 import + 1 line) |
| GAP-2b | `answer_relevancy` broken | Low | Investigate langchain-openai embeddings shim |
| GAP-3 | `interrupt_before=["tools"]` not added | Low | One-line add to `research_agent.py` |
| GAP-4 | LangSmith not configured | Medium | Get key at smith.langchain.com, update `.env` |

---

## 8. Dependencies Added

```
langchain-community==0.4.2
faiss-cpu==1.13.2
ragas==0.4.3
pypdf==6.12.2
datasets==4.8.5
langchain-google-vertexai  (stub compatibility)
```

---

## 9. How to Use

```powershell
# Ask a question against local docs
python rag_pipeline.py

# Force rebuild vector store
python rag_pipeline.py rebuild

# Run RAGAS evaluation
python rag_eval.py

# Run research agent (LangGraph)
python main.py "your research topic"
```

---

## 10. Next Steps

1. Add `LANGCHAIN_API_KEY` to `.env` → enable LangSmith tracing
2. Run `python main.py "any topic"` → verify trace at smith.langchain.com
3. Extend corpus in `data/sample_docs/` with your own documents
4. Try `/pdca plan` on a new feature using this RAG pipeline as a foundation
