import os
import sys
import uuid

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from graph import graph
from config import MODEL_NAME, OPENAI_API_KEY

# ── LLM for judging — temperature=0 for consistent scoring ────
judge_llm = ChatOpenAI(
    model = MODEL_NAME,
    api_key = OPENAI_API_KEY,
    temperature = 0
)

# ---------------------------------------
# STRATEGY 1 — Exact match checks
# ---------------------------------------

def check_verdict_valid(result: dict) -> dict:
    """Critic verdict must be exactly 'approved' or 'needs_revision'."""
    verdict = result.get("verdict", "").strip().lower()
    passed = verdict in {"approved", "needs_revision"}

    return {
        "check": "verdict_valid",
        "passed": passed,
        "detail": f"verdict = '{verdict}'"
    }

def check_revision_count(result: dict) -> dict:
    """Revision count must be between 0 and 2."""
    count = result.get("revision_count", -1)
    passed = 0 <= count <= 2

    return {
        "check": "revision_count",
        "passed": passed,
        "detail": f"count = {count}"
    }

def check_all_fields_present(result: dict) -> dict:
    """All required state fields must be populated."""
    required = ["task", "research_output", "draft_report",
                "final_report", "verdict"]
    missing = [f for f in required if not result.get(f)]
    passed = len(missing) == 0
    return {
        "check":  "all_fields_present",
        "passed": passed,
        "detail": f"missing={missing}"
    }

# ---------------------------------------
# STRATEGY 2 — Criteria-based checks
# ---------------------------------------
def check_report_sections(result: dict) -> dict:
    """Final report must contain Introduction, Key Findings, Conclusion."""
    report = result.get("final_report", "").lower()
    sections = ["introduction", "key findings", "conclusion"]
    missing = [s for s in sections if s not in report]
    passed = len(missing) == 0
    return {
        "check": "report_sections",
        "passed": passed,
        "detail": f"missing={missing}"
    }

def check_report_length(result: dict) -> dict:
    """Final report must be at least 300 words."""
    report = result.get("final_report", "")
    words  = len(report.split())
    passed = words >= 300
    return {
        "check":  "report_length",
        "passed": passed,
        "detail": f"words={words} (min=300)"
    }

def check_research_depth(result: dict) -> dict:
    """Research output must have at least 50 words."""
    research = result.get("research_output", "")
    words    = len(research.split())
    passed   = words >= 50
    return {
        "check":  "research_depth",
        "passed": passed,
        "detail": f"words={words} (min=50)"
    }

# ---------------------------------------
# STRATEGY 3 — LLM-as-judge
# ---------------------------------------

def check_llm_judge(result: dict) -> dict:
    """Use LLM to score report quality on a 1-5 scale."""
    report = result.get("final_report", "")
    topic = result.get("task", "")

    prompt = f"""Rate the following report on a scale 1 to 5.

1 = Very poor — missing sections, inaccurate, unclear
2 = Poor — incomplete or poorly structured
3 = Acceptable — covers basics but lacks depth
4 = Good — well structured, accurate, clear
5 = Excellent — comprehensive, well-written, insightful

TOPIC: {topic}

REPORT:
{report}

Respond in EXACTLY this format:
SCORE: [number]
REASON: [one sentence]"""

    response = judge_llm.invoke(prompt)
    raw = response.content

    # Parse Score
    score = 3 # default
    reason = raw

    try:
        if "SCORE:" in raw:
            score = int(raw.split("SCORE:")[1].split("REASON:")[0].strip())
            reason = raw.split("REASON:")[1].strip()
    except Exception:
        pass

    passed = score >=3

    return {
        "check":  "llm_judge",
        "passed": passed,
        "detail": f"score={score}/5 | {reason}"
    }

# ---------------------------------------
# All Checks
# ---------------------------------------

ALL_CHECKS = [
    check_verdict_valid,
    check_revision_count,
    check_all_fields_present,
    check_report_sections,
    check_report_length,
    check_research_depth,
    check_llm_judge,
]

# ---------------------------------------
# Test Cases
# ---------------------------------------

TEST_CASES = [
    {
        "name":  "langgraph_features",
        "input": "Write a report on the key features of LangGraph",
    },
    {
        "name":  "python_data_science",
        "input": "Write a report on why Python is popular for data science",
    },
    {
        "name":  "rag_overview",
        "input": "Write a report on RAG",
    },
]

# ---------------------------------------
# Eval Runner
# ---------------------------------------

def run_single_eval(name: str, user_input: str) -> dict:
    """Run pipeline + all checks for one test case."""
    print(f"\n{'─'*50}")
    print(f"Test: {name}")
    print(f"Input: {user_input}")
    print(f"{'─'*50}")

    # Run pipeline
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content = user_input)]},
            config = {"configurable": {
                "thread_id": f"eval_{uuid.uuid4().hex[:8]}"
            }}
        )
    except Exception as e:
        print(f"  Pipeline CRASHED: {e}")
        return {"name": name, "error": str(e), "checks": []}
    
    # Run all checks
    check_results = []
    for check_fn in ALL_CHECKS:
        try:
            cr = check_fn(result)
            check_results.append(cr)
            status = "PASS" if cr["passed"] else "FAIL"
            print(f"  [{status}] {cr['check']}: {cr['detail']}")
        except Exception as e:
            print(f"  [ERROR] {check_fn.__name__}: {e}")

    passed = sum(1 for cr in check_results if cr["passed"])
    total = len(check_results)
    print(f"\n  Result: {passed}/{total} checks passed")

    return {
        "name":   name,
        "passed": passed,
        "total":  total,
        "checks": check_results
    }

def run_all_evals():
    """Run evaluation across all test cases."""
    print(f"\n{'='*50}")
    print("AGENT EVALUATION")
    print(f"{'='*50}")

    results = []
    for tc in TEST_CASES:
        r = run_single_eval(tc["name"], tc["input"])
        results.append(r)

    # Summary
    print(f"\n{'='*50}")
    print("EVAL SUMMARY")
    print(f"{'='*50}")

    for r in results:
        if "error" in r:
            print(f"  {r['name']:<25} PIPELINE ERROR")
        else:
            print(f"  {r['name']:<25} {r['passed']}/{r['total']} passed")

    total_passed = sum(r.get("passed", 0) for r in results)
    total_checks = sum(r.get("total", 0) for r in results)
    print(f"\n  Total: {total_passed}/{total_checks} checks passed")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_all_evals()