from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic.
    Use for any factual question or recent news.
    Do NOT use for math calculations."""
    
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results = 3))
        if not results:
            return "No results found"
        return "\n".join(
            f"- {r['title']}: {r['body']}"
            for r in results
        )
    except Exception as e:
        return f"Search failed: {e}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression.
    Use for any arithmetic or numeric calculation.
    E.g. '1234*5678 or 15/100 * 4750'."""

    try:
        import math
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression, {__builtins__: {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {e}"



# ── Tool registry ──────────────────────────────────────────────
tools = [web_search, calculator]
TOOL_MAP = {t.name: t for t in tools}