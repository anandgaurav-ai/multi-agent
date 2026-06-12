from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from config import MODEL_NAME, OPENAI_API_KEY
from config import RESEARCHER_PROMPT, WRITER_PROMPT, CRITIC_PROMPT
from tools import tools, TOOL_MAP
import json

# ── Shared LLM ─────────────────────────────────────────────────
llm = ChatOpenAI(model = MODEL_NAME, api_key = OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools)


# ── Internal agent runner ──────────────────────────────────────
def run_subagent(system_prompt: str, task: str, use_tools: bool = False) -> str:
    """
    Run a single subagent with its own internal loop.
    Returns the final text response.
    """

    messages = [
        SystemMessage(content = system_prompt),
        HumanMessage(content = task)
    ]

    # Research agent uses tools, writer and critic don't
    active_llm = llm_with_tools if use_tools else llm

    print(f"\n [subagent running] task: {task[:80]}")

    for step in range(10):
        response = active_llm.invoke(messages)
        messages.append(response)

        # If tool calls - execute them an dcontinue loop
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                print(f"    -> {name} ({args})")

                if name not in TOOL_MAP:
                    result = f"Error: tool '{name}' not found."
                else:
                    result = TOOL_MAP[name].invoke(args)
                    result = result if result is not None else "No output."

                messages.append(ToolMessage(
                    content = str(result),
                    tool_call_id = tool_call["id"]
                ))

        else:
            # No tool calls - Final answer
            return response.content
        
    return "Max steps reached."


# ── Subagent functions ─────────────────────────────────────────

def research_agent(task: str) -> str:
    """ Search the web and gather facts on the given topic."""
    print("\n Research Agent starting...")

    result = run_subagent(
        system_prompt = RESEARCHER_PROMPT,
        task = task,
        use_tools = True
    )
    print("Research Agent done")
    return result

def writer_agent(task: str, research: str) -> str:
    """Write a structured report based on research findings."""
    print("\n Writer agent starting...")
    full_task = f"""Write a structured report on the following topic.

TOPIC: {task}

RESEARCH_FINDINGS:
{research}

write a prefessional report with Introduction, Key Findings, and Conclusion."""
    
    result = run_subagent(
        system_prompt = WRITER_PROMPT,
        task = full_task,
        use_tools = False
    )
    print("Writer Agent Done")
    return result

def critic_agent(draft: str) -> str:
    """Review the draft report. Returns verdict, feedback, and improved report."""
    print("\n Critic Agent starting...")
    full_task = f"""Review the following report.

    
DRAFT_REPORT:
{draft}"""
    
    raw_result = run_subagent(
        system_prompt = CRITIC_PROMPT,
        task = full_task,
        use_tools = False
    )
    
    # ── Parse the structured output ─────────────────────────
    verdict = "needs_revision" # safe default
    feedback = ""
    report = raw_result # fallback — whole thing if parsing fails

    try:
        if "VERDICT:" in raw_result and "REPORT:" in raw_result:
            verdict_part = raw_result.split("VERDICT:")[1].split("FEEDBACK:")[0].strip()
            feedback_part = raw_result.split("FEEDBACK:")[1].split("REPORT:")[0].strip()
            report_part = raw_result.split("REPORT:")[1].strip()

            verdict = verdict_part.lower()
            feedback = feedback_part
            report = report_part
    except Exception as e:
        print(f"Parsing failed, using raw output: {e}")
    
    
    print(f" Verdict: {verdict}")
    print(" Critic Agent done.")
    
    return {
        "verdict": verdict,
        "feedback": feedback,
        "report": report
    }