import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
from config import DB_PATH, OPENAI_API_KEY
from state import OrchestratorState
from agents import research_agent, writer_agent, critic_agent

MAX_REVISIONS = 2

# ── Nodes ──────────────────────────────────────────────────────

def orchestrator_node(state: OrchestratorState) -> dict:
    """Extract task from last Human message and initialise state."""
    last_message = state["messages"][-1]
    task = last_message.content
    print(f"\n Orchestrator: task received - '{task[:80]}...'")
    return {
        "task": task,
        "revision_count": 0 # Initialise counter
        }

def research_node(state: OrchestratorState) -> dict:
    """Run research agent on the task"""
    result = research_agent(state["task"])
    print(f"\n Research output preview: {result[:150]}...")
    return {"research_output": result}

def writer_node(state: OrchestratorState) -> dict:
    """Run writer agent - first pass or revision based on feedback."""
    feedback = state.get("feedback", "") # empty on first pass
    

    result = writer_agent(
        task = state["task"],
        research = state["research_output"],
        feedback = feedback if feedback else None
    )
    print(f"\n Draft preview: {result[:150]}...")
    return {"draft_report": result}

def critic_node(state: OrchestratorState) -> dict:
    """Run critic agent - Returns verdict, feedback, and report."""
    result = critic_agent(state["draft_report"])

    new_revision_count = state["revision_count"]
    if result["verdict"] == "needs_revision":
        new_revision_count +=1

    print(f"Final report preview: {result['report'][:150]}...")
    return {
        "final_report": result["report"],
        "verdict": result["verdict"],
        "feedback": result["feedback"],
        "revision_count": new_revision_count
        }

# ── Conditional routing ─────────────────────────────────────────

def should_revise(state: OrchestratorState) -> str:
    """Decide whether to loop back to writer or finish."""
    if state["verdict"] == "approved":
        print("\n Verdict: APPROVED - Pipeline complete")
        return "end"
    
    if state["revision_count"] >= MAX_REVISIONS:
        print(f"\n Max revisions ({MAX_REVISIONS}) reached - accepting current report")
        return "end"
    
    print(f"\n Verdict: NEED_REVISION (revision {state['revision_count']}/{MAX_REVISIONS}) - back to writer")
    return "revise"


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
    
    # Conditional edge — the feedback loop
    builder.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise": "writer", # loop back
            "end": END          # Finish
        }
    )

    return builder.compile(checkpointer = checkpointer)

# ── Database connection ────────────────────────────────────────

conn = sqlite3.connect(DB_PATH, check_same_thread = False)
checkpointer = SqliteSaver(conn)
graph = build_graph(checkpointer = checkpointer)