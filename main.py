import os
import sys
import uuid


from langchain_core.messages import HumanMessage
from graph import graph

def run_report_pipeline(topic: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'='*60}")
    print(f"Task: {topic}")
    print(f"\n{'='*60}")

    result = graph.invoke(
        {"messages": [HumanMessage(content = topic)]},
        config = config
    )

    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}")
    print(result["final_report"])

    return result

if __name__ == "__main__":
    run_report_pipeline(
        "Write a report on the key features of LangGraph",
        thread_id=f"report_{uuid.uuid4().hex[:8]}"
    )