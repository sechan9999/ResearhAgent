"""
main.py  —  Run the Research Agent
====================================
Usage:
    python main.py "Your research topic here"
    python main.py          # uses the default example topic

LangSmith tracing is automatically enabled when these env vars are set:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=<your key>
    LANGCHAIN_PROJECT=research-agent
"""

import sys
from dotenv import load_dotenv

load_dotenv()

# LangSmith: just setting env vars is enough — LangChain picks them up automatically.
# No extra code needed beyond what's in .env

from research_agent import run_research

DEFAULT_TOPIC = "What are the latest breakthroughs in large language model efficiency (2024-2025)?"


def main():
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TOPIC
    print(f"\nResearching: {topic}\n{'='*60}\n")

    report = run_research(topic)

    print(report)
    print("\n" + "="*60)
    print("Done.")


if __name__ == "__main__":
    main()
