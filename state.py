from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class OrchestratorState(TypedDict):
    messages:           Annotated[list, add_messages]
    task:               str # original user task
    research_output:    str # output from research agent
    draft_report:       str # output from writer agent
    final_report:       str # output from critic agent

    # ── New fields for feedback loop ────────────────────────

    verdict:            str # "approved" or "needs_revision"
    feedback:           str # Critic's feedback for writer
    revision_count:     int # tracks number of revisions made