# Gap Analysis: langchain-rag-learning

**Date**: 2026-06-02  
**Phase**: Check  
**Design Ref**: `docs/02-design/features/langchain-rag-learning.design.md`

---

## Match Rate Summary

| Category | Implemented | Total | Score |
|----------|-------------|-------|-------|
| File Structure | 7 | 7 | 100% |
| rag_pipeline.py API | 6.5 | 7 | 93% |
| rag_eval.py metrics | 2 | 4 | 50% |
| research_agent.py improvements | 1 | 2 | 50% |
| Dependencies | 5 | 5 | 100% |
| Environment Variables (LangSmith) | 0 | 3 | 0%* |
| Acceptance Criteria | 3 | 4 | 75% |
| **Overall** | **24.5** | **32** | **77%** |

> *LangSmith env vars blocked by user constraint (no LANGCHAIN_API_KEY). Adjusted match rate excluding these 4 items: **88%**.

---

## Implemented (Matches Design)

- [x] `rag_pipeline.py` created with all 4 public functions
- [x] FAISS vector store with `text-embedding-3-small`, chunk=500, overlap=50
- [x] LCEL chain: `retriever | format_docs | prompt | llm | StrOutputParser`
- [x] Prompt matches design: "Answer using only the context below. Context: {context}. Question: {question}"
- [x] `data/sample_docs/` — 4 corpus files (langchain, langgraph, ragas, langsmith)
- [x] `data/faiss_index/` — persisted (index.faiss + index.pkl)
- [x] `rag_eval.py` with `build_eval_dataset()` and `run_ragas()`
- [x] `faithfulness` and `context_precision` metrics running
- [x] RAGAS scores: faithfulness=0.917, context_precision=0.958 (both > 0.7 threshold)
- [x] `research_agent.py` — `MemorySaver` checkpointer added at `builder.compile()`
- [x] All new dependencies installed and in `requirements.txt`

---

## Gaps

### GAP-1: `ask()` signature mismatch (Minor)
- **Design**: `ask(chain, question: str) -> dict`
- **Actual**: `ask(chain, retriever, question: str) -> dict`
- **Impact**: Low — `retriever` is needed to return source docs; design was incomplete
- **Fix**: Update design doc to reflect actual signature (preferred), or refactor to hide retriever internally

### GAP-2: RAGAS metrics incomplete (Medium)
- **Design**: 4 metrics — `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`
- **Actual**: 2 metrics — `faithfulness`, `context_precision`
- **Reason**: `answer_relevancy` requires `embed_query` (removed in langchain-openai 1.x); `context_recall` not implemented
- **Impact**: Medium — evaluation coverage reduced; core faithfulness score still valid
- **Fix**: Add `context_recall` (has ground_truth data already); investigate `answer_relevancy` workaround

### GAP-3: `interrupt_before=["tools"]` not implemented (Minor)
- **Design**: Add human-in-loop option via `interrupt_before=["tools"]` to `builder.compile()`
- **Actual**: Only `MemorySaver` added; no interrupt
- **Impact**: Low — optional feature; does not break existing behavior
- **Fix**: One-line addition to `research_agent.py`

### GAP-4: LangSmith not configured (External constraint)
- **Design**: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` in `.env`
- **Actual**: `LANGCHAIN_TRACING_V2=false`, no API key
- **Reason**: User has not obtained `LANGCHAIN_API_KEY` yet
- **Impact**: Acceptance criterion 4 not met; no production observability
- **Fix**: User signs up at smith.langchain.com and adds key to `.env`

---

## RAGAS Evaluation Results

| Question | Faithfulness | Context Precision |
|----------|-------------|-------------------|
| What is LCEL and how does it work? | 0.667 | 1.000 |
| What env vars does LangSmith need? | 1.000 | 1.000 |
| How does RAGAS measure faithfulness? | 1.000 | 0.833 |
| What is a MemorySaver in LangGraph? | 1.000 | 1.000 |
| **Average** | **0.917** | **0.958** |

---

## Recommendation

**Adjusted match rate (excluding LangSmith external constraint): 88%**

Close to the 90% threshold. Two quick fixes close the gap:
1. Add `context_recall` metric to `rag_eval.py` (+1 metric, partial restore of GAP-2)
2. Add `interrupt_before=["tools"]` to `research_agent.py` (one line, closes GAP-3)

LangSmith (GAP-4) is pending user action and should not block progress.
