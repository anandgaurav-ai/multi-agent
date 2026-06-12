import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
from config import DB_PATH, OPENAI_API_KEY
from state import OrchestratorState
from agents import research_agent, writer_agent, critic_agent

# ── Nodes ──────────────────────────────────────────────────────

def orchestrator_node(state: OrchestratorState) -> dict:
    """Extract task from last Human message and store in state."""
    last_message = state["messages"][-1]
    task = last_message.content
    print(f"\n Orchestrator: task received - '{task[:80]}...'")
    return {"task": task}

def research_node(state: OrchestratorState) -> dict:
    """Run research agent on the task"""
    result = research_agent(state["task"])
    print(f"\n Research output preview: {result[:150]}...")
    return {"research output": result}

def writer_node(state: OrchestratorState) -> dict:
    "Run writer agent with task + research output."
    result = writer_agent(
        task = state["task"],
        research = state["research_output"]
    )
    print(f"\n Draft preview: {result[:150]}...")
    return {"draft_report": result}

def critic_node(state: OrchestratorState) -> dict:
    "Run critic agent on the draft report."
    result = critic_agent(state["draft_report"])
    print(f"Final report preview: {result[:150]}...")
    return {"final_report": result}


# ── Graph builder ──────────────────────────────────────────────

def build_graph(checkpointer = None):
    builder = StateGraph(OrchestratorState)


    # Add nodes
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("researcher", research_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)

    # Add edges — fixed sequential pipeline
    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "critic")
    builder.add_edge("critic", END)

    return builder.compile(checkpointer = checkpointer)

# ── Database connection ────────────────────────────────────────

conn = sqlite3.connect(DB_PATH, check_same_thread = False)
checkpointer = SqliteSaver(conn)
graph = build_graph(checkpointer = checkpointer)
