# Project 2: LangGraph Research Agent

A LangGraph-powered agent that searches the web, reads pages, and synthesises a structured report — with memory to avoid re-fetching sources and LangSmith tracing for full observability.

## Architecture

```
User query
    │
    ▼
┌─────────┐     tool_calls?     ┌───────┐
│  agent  │ ─────── yes ──────▶ │ tools │
│  (LLM)  │ ◀──────────────────  └───────┘
└─────────┘       loop               │
    │                          ┌─────▼──────┐
    │ no tool_calls            │ track_urls │  ← updates URL cache in state
    ▼                          └────────────┘
  END (report)
```

## Tools

| Tool | Description |
|---|---|
| `web_search` | Searches the web via [Tavily API](https://tavily.com) |
| `read_url` | Fetches and extracts text from a URL; results are **cached** to avoid re-fetching |
| `python_repl` | Executes Python code for calculations or data processing |

## Memory

`read_url` uses an in-session dict cache (`_url_cache`). If the agent tries to read a URL it already fetched, it gets the cached text instantly — no duplicate HTTP requests.

## Tracing (LangSmith)

Every node, tool call, and LLM invocation is automatically traced. Set these env vars and open [smith.langchain.com](https://smith.langchain.com) to inspect runs:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=research-agent
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys
```

`.env` keys needed:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
LANGCHAIN_API_KEY=ls__...   # optional, for tracing
```

## Usage

```bash
# Default topic (LLM efficiency breakthroughs)
python main.py

# Custom topic
python main.py "Impact of AI on drug discovery 2025"
```

The agent will search, read sources, optionally run Python, then print a cited report with a `### Summary` section.

## Skills Covered

- **LangGraph** — `StateGraph` with conditional routing and a tool-execution loop
- **Tool use** — custom `@tool` functions bound to the LLM
- **Agent state** — `TypedDict` state with `add_messages` reducer
- **Memory** — URL cache prevents redundant fetches within a run
- **Observability** — LangSmith tracing with zero extra code

## Project Structure

```
.
├── research_agent.py   # Core agent: state, tools, graph
├── main.py             # CLI entrypoint
├── requirements.txt
└── .env.example
```
