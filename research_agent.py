"""
research_agent.py  —  LangGraph Research Agent
================================================
Tools:
  - web_search   : Tavily API
  - read_url     : scrape page text (cached per run)
  - python_repl  : run arbitrary Python

Memory:
  - _url_cache (dict) prevents re-fetching the same URL

Tracing:
  - Set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY in .env
    LangChain picks these up automatically — no extra code needed.
"""

import os
import re
import io
import contextlib
from typing import Annotated, TypedDict

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from tavily import TavilyClient

load_dotenv()

# ──────────────────────────────────────────────
# 1. IN-SESSION URL CACHE  (memory)
# ──────────────────────────────────────────────
_url_cache: dict = {}   # url -> extracted text


# ──────────────────────────────────────────────
# 2. TOOLS
# ──────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web with Tavily. Returns titles, URLs, and snippets."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(query=query, max_results=5)
    parts = []
    for r in results.get("results", []):
        parts.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}"
        )
    return "\n\n---\n\n".join(parts) if parts else "No results found."


@tool
def read_url(url: str) -> str:
    """
    Fetch and extract readable text from a URL.
    Repeated calls with the same URL return the cached version — no re-fetch.
    """
    if url in _url_cache:
        return "[CACHED]\n" + _url_cache[url]

    try:
        headers = {"User-Agent": "Mozilla/5.0 (research-agent/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()[:4000]
        _url_cache[url] = text
        return text
    except Exception as e:
        return f"Error fetching {url}: {e}"


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return stdout. Use for calculations or data processing."""
    buf = io.StringIO()
    local_vars: dict = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<repl>", "exec"), local_vars)
        output = buf.getvalue()
        # capture last expression value
        last_line = code.strip().split("\n")[-1]
        try:
            val = eval(last_line, local_vars)
            if val is not None:
                output += f"\n=> {val}"
        except Exception:
            pass
        return output.strip() or "(no output)"
    except Exception as e:
        return f"Error: {e}"


# ──────────────────────────────────────────────
# 3. AGENT STATE
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    visited_urls: list


# ──────────────────────────────────────────────
# 4. LLM + TOOLS
# ──────────────────────────────────────────────

TOOLS = [web_search, read_url, python_repl]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are an expert research agent. Your job:
1. Search the web to find relevant sources on the user's topic.
2. Read the most promising URLs to extract details.
3. Use the Python REPL for any calculations or data processing.
4. Synthesise everything into a structured, well-cited report.

Rules:
- Always cite sources with URLs.
- Do NOT re-fetch a URL you have already read (read_url caches results).
- Try alternative search queries if initial results are thin.
- End with a clear ### Summary section.
"""


# ──────────────────────────────────────────────
# 5. GRAPH NODES
# ──────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """LLM chooses the next tool call or produces the final answer."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def track_urls(state: AgentState) -> dict:
    """Record newly cached URLs into state after each tool execution."""
    return {"visited_urls": list(_url_cache.keys())}


def should_continue(state: AgentState) -> str:
    """Routing: tool_calls present -> run tools, else -> done."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ──────────────────────────────────────────────
# 6. BUILD GRAPH
# ──────────────────────────────────────────────

tool_node = ToolNode(TOOLS)

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("track", track_urls)

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "track")
builder.add_edge("track", "agent")

graph = builder.compile(checkpointer=MemorySaver())


# ──────────────────────────────────────────────
# 7. PUBLIC API
# ──────────────────────────────────────────────

def run_research(topic: str) -> str:
    """Run the research agent on a topic and return the final report string."""
    _url_cache.clear()

    initial: AgentState = {
        "messages": [HumanMessage(
            content=f"Research this topic and produce a detailed report:\n\n{topic}"
        )],
        "visited_urls": [],
    }

    final = graph.invoke(initial, config={"configurable": {"thread_id": "research-1"}})
    report = final["messages"][-1].content

    visited = final.get("visited_urls", [])
    if visited:
        report += (
            f"\n\n---\n**Sources fetched ({len(visited)}):**\n"
            + "\n".join(f"- {u}" for u in visited)
        )

    return report
