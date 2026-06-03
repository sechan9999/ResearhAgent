"""
rag_eval.py  —  RAGAS Evaluation for the RAG Pipeline
=======================================================
Usage:
    python rag_eval.py

Requires:
    - Vector store already built (run rag_pipeline.py first)
    - OPENAI_API_KEY in .env
"""

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision

from rag_pipeline import load_vectorstore, build_rag_chain, ask

EVAL_QUESTIONS = [
    {
        "question": "What is LCEL and how does it work?",
        "ground_truth": "LCEL (LangChain Expression Language) is a way to compose chains using the pipe operator to connect components like retriever, prompt, LLM, and output parser.",
    },
    {
        "question": "What environment variables does LangSmith need?",
        "ground_truth": "LangSmith requires LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY, and LANGCHAIN_PROJECT environment variables.",
    },
    {
        "question": "How does RAGAS measure faithfulness?",
        "ground_truth": "RAGAS measures faithfulness by checking whether every claim in the answer is supported by the retrieved context. A score of 1.0 means fully grounded.",
    },
    {
        "question": "What is a MemorySaver in LangGraph?",
        "ground_truth": "MemorySaver is a built-in checkpointer in LangGraph that stores graph state in memory, allowing threads to maintain context across multiple invocations.",
    },
]


def build_eval_dataset(chain, retriever) -> Dataset:
    rows = []
    for item in EVAL_QUESTIONS:
        result = ask(chain, retriever, item["question"])
        source_docs = retriever.invoke(item["question"])
        rows.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": [d.page_content for d in source_docs],
            "ground_truth": item["ground_truth"],
        })
    return Dataset.from_list(rows)


def run_ragas(dataset: Dataset):
    result = evaluate(dataset, metrics=[faithfulness, context_precision])
    return result.to_pandas()


if __name__ == "__main__":
    print("Loading vector store...")
    vs = load_vectorstore()
    chain, retriever = build_rag_chain(vs)

    print("Building evaluation dataset (4 questions)...")
    dataset = build_eval_dataset(chain, retriever)

    print("Running RAGAS evaluation...")
    scores = run_ragas(dataset)

    print(f"\nColumns: {scores.columns.tolist()}")
    print("\n=== RAGAS Scores ===")
    metric_cols = [c for c in scores.columns if c in ("faithfulness", "context_precision")]
    id_col = next((c for c in ("question", "user_input") if c in scores.columns), None)
    show_cols = ([id_col] if id_col else []) + metric_cols
    print(scores[show_cols].to_string(index=False))

    print("\n=== Averages ===")
    for col in metric_cols:
        print(f"  {col}: {scores[col].mean():.3f}")
