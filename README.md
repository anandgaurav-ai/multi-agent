# Multi-Agent Research Report Pipeline

A production-style multi-agent system that researches, writes, and reviews reports with automated quality control.

## Features
- Three specialised agents: Researcher, Writer, Critic
- Feedback loop — critic can send reports back for revision
- Quality gates with max revision limits
- SQLite checkpointing for conversation persistence
- LangSmith tracing for full observability
- Automated evaluation harness with 3 strategies

## Architecture
- **Orchestrator** — extracts task, initialises state
- **Research Agent** — searches web, gathers facts
- **Writer Agent** — writes structured reports from research
- **Critic Agent** — reviews, gives verdict + feedback
- **Conditional routing** — loops back to writer if needs revision

## Stack
LangGraph · LangChain · OpenAI GPT-4o · SQLite · DuckDuckGo · LangSmith

## Setup
pip install -r requirements.txt
cp .env.example .env  # add your API keys
python main.py

## Evaluation
python eval.py

Runs 3 test cases with 7 checks each (21 total):

**Exact match:** verdict valid, revision count, all fields present

**Criteria:** report sections, minimum length, research depth

**LLM-as-judge:** GPT-4o scores quality 1-5

## Project Structure
config.py   — API keys, constants, agent prompts
state.py    — OrchestratorState definition
tools.py    — web_search, calculator
agents.py   — research, writer, critic subagents
graph.py    — orchestrator graph with feedback loop
main.py     — entry point
eval.py     — evaluation harness