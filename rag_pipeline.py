"""
rag_pipeline.py  —  LangChain RAG Pipeline
============================================
Usage:
    python rag_pipeline.py          # build index + answer a sample question
    python rag_pipeline.py rebuild  # force rebuild the vector store

Public API:
    build_vectorstore(docs_dir)  -> FAISS
    load_vectorstore()           -> FAISS
    build_rag_chain(vectorstore) -> Runnable
    ask(chain, question)         -> dict {answer, sources}
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DOCS_DIR = Path("data/sample_docs")
INDEX_DIR = Path("data/faiss_index")

PROMPT = ChatPromptTemplate.from_template(
    "Answer using only the context below. If unsure, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def build_vectorstore(docs_dir: str | Path = DOCS_DIR) -> FAISS:
    loader = DirectoryLoader(str(docs_dir), glob="*.txt", loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Vector store built: {len(chunks)} chunks from {len(docs)} docs -> {INDEX_DIR}")
    return vectorstore


def load_vectorstore() -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


def build_rag_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | StrOutputParser()
    )
    return chain, retriever


def ask(chain, retriever, question: str) -> dict:
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    sources = list({d.metadata.get("source", "unknown") for d in source_docs})
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    rebuild = len(sys.argv) > 1 and sys.argv[1] == "rebuild"

    if rebuild or not (INDEX_DIR / "index.faiss").exists():
        vs = build_vectorstore()
    else:
        vs = load_vectorstore()
        print(f"Loaded existing vector store from {INDEX_DIR}")

    chain, retriever = build_rag_chain(vs)

    questions = [
        "What is LCEL and how does it work?",
        "How does RAGAS measure faithfulness?",
        "What environment variables does LangSmith need?",
        "What is a MemorySaver in LangGraph?",
    ]

    for q in questions:
        result = ask(chain, retriever, q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 60)
