# Design: LangChain RAG Learning

**Feature**: `langchain-rag-learning`  
**Phase**: Design  
**Created**: 2026-06-02  
**Ref Plan**: `docs/01-plan/features/langchain-rag-learning.plan.md`

---

## 1. Architecture Overview

```
docs/ (corpus)
   └─ *.txt / *.pdf
          │
          ▼
   DocumentLoader          (LangChain: TextLoader / PyPDFLoader)
          │
          ▼
   TextSplitter             (RecursiveCharacterTextSplitter, chunk=500, overlap=50)
          │
          ▼
   Embeddings               (OpenAIEmbeddings: text-embedding-3-small)
          │
          ▼
   VectorStore              (FAISS, persisted to data/faiss_index/)
          │
   ┌──────┴──────┐
   │             │
retriever     rag_eval.py
(k=4)            │
   │          RAGAS metrics
   ▼             │
   RAG Chain     ▼
(LangChain       faithfulness
 LCEL pipe)      answer_relevancy
   │             context_recall
   ▼             context_precision
 Answer
   │
   ▼
LangSmith trace (if LANGCHAIN_API_KEY set)
```

---

## 2. File Structure

```
Langchain/
├── main.py                      (existing — research agent runner)
├── research_agent.py            (existing — LangGraph agent)
├── rag_pipeline.py              (NEW — RAG Q&A pipeline)
├── rag_eval.py                  (NEW — RAGAS evaluation)
├── data/
│   ├── sample_docs/             (NEW — corpus: .txt or .pdf files)
│   └── faiss_index/             (NEW — persisted vector store)
├── requirements.txt             (UPDATE — add new deps)
└── .env                         (UPDATE — add LANGCHAIN_API_KEY)
```

---

## 3. Module Design

### 3.1 `rag_pipeline.py`

```python
# Public API
build_vectorstore(docs_dir: str) -> FAISS          # load, split, embed, save
load_vectorstore() -> FAISS                         # load persisted index
build_rag_chain(vectorstore: FAISS) -> Runnable     # retriever + prompt + llm
ask(chain, question: str) -> dict                   # returns {answer, source_docs}
```

**RAG chain (LCEL):**
```
retriever | format_docs | prompt | llm | StrOutputParser
```

**Prompt template:**
```
Answer using only the context below.
Context: {context}
Question: {question}
```

### 3.2 `rag_eval.py`

```python
# Public API
build_eval_dataset(chain, questions: list[str]) -> Dataset   # generate answers + retrieve contexts
run_ragas(dataset: Dataset) -> pd.DataFrame                  # scores per question
```

**RAGAS metrics used:**
| Metric | Measures |
|--------|----------|
| `faithfulness` | Answer grounded in context? |
| `answer_relevancy` | Answer relevant to question? |
| `context_recall` | Context covers the ground truth? |
| `context_precision` | Context is precise (no noise)? |

> Note: `context_recall` requires `ground_truth` — use synthetic or manually written ground truths for the sample dataset.

### 3.3 `research_agent.py` — LangGraph improvements

Two targeted additions (no rewrite):

| Addition | Detail |
|----------|--------|
| Memory checkpointing | Add `MemorySaver` checkpointer to `builder.compile()` |
| Interrupt-before-tools | Add `interrupt_before=["tools"]` for human-in-loop option |

---

## 4. Dependencies

```
# requirements.txt additions
langchain-community>=0.3.0     # document loaders, FAISS wrapper
faiss-cpu>=1.8.0               # vector store
ragas>=0.2.0                   # RAG evaluation
langsmith>=0.1.0               # tracing SDK (already listed, confirm version)
pypdf>=4.0.0                   # PDF loader (optional)
```

---

## 5. Environment Variables

```ini
# .env additions
LANGCHAIN_TRACING_V2=true          # enable LangSmith (change from false)
LANGCHAIN_API_KEY=ls__...          # from smith.langchain.com
LANGCHAIN_PROJECT=research-agent   # project name in LangSmith UI
```

---

## 6. Implementation Order

1. Install new dependencies
2. Add sample docs to `data/sample_docs/`
3. Implement `rag_pipeline.py` — build & persist vectorstore, build chain, `ask()`
4. Smoke test: `python rag_pipeline.py`
5. Implement `rag_eval.py` — dataset builder + RAGAS runner
6. Smoke test: `python rag_eval.py`
7. Update `research_agent.py` — add checkpointer + interrupt option
8. Add `LANGCHAIN_API_KEY` to `.env`, set `LANGCHAIN_TRACING_V2=true`
9. Run `main.py` and verify trace appears in LangSmith UI

---

## 7. Acceptance Criteria

- [ ] `rag_pipeline.py` answers questions from local docs with source citations
- [ ] `rag_eval.py` produces a RAGAS score table (faithfulness >= 0.7)
- [ ] `research_agent.py` compiles with `MemorySaver` without breaking existing behavior
- [ ] At least one LangSmith trace visible in `smith.langchain.com` under project `research-agent`
