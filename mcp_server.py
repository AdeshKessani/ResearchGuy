"""
MCP server exposing the research agent pipeline as a callable tool.

This is a thin wrapper -- it doesn't duplicate any pipeline logic, it
just calls agent.pipeline.run_pipeline(), the same function main.py
and eval/run_eval.py already use. 

COST WARNING!!!
"""

import time

from mcp.server.fastmcp import FastMCP

from agent.config import load_settings
from agent.pipeline import run_pipeline
from agent.trace import Tracer

mcp = FastMCP("research-agent")

# Settings loaded once per server process, not per call -- avoids
# re-reading and re-validating environment variables on every request.
_settings = None


def _get_settings():
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


@mcp.tool()
def research_question(question: str) -> str:
    """
    Run the full autonomous research pipeline on a question and return
    a cited, self-critiqued report as markdown.

    This runs Planner -> Researcher -> Synthesizer -> Critic/Reviser
    (up to 2 revision passes) end to end. It makes multiple LLM and web
    search API calls and typically takes well over a minute -- this is
    a genuine research task, not a quick lookup, and should be called
    accordingly.

    Args:
        question: A broad research question to investigate.

    Returns:
        A markdown report with inline [n] citations, a source list,
        and any known gaps the pipeline couldn't fully resolve after
        its revision passes.
    """
    settings = _get_settings()
    run_id = f"mcp_{int(time.time())}"
    tracer = Tracer(run_id)

    final = run_pipeline(question, settings, tracer, verbose=False)

    lines = [f"# {final.title}", "", final.summary, "", final.body_markdown, ""]

    if final.known_gaps:
        lines.append(f"## Known gaps (after {final.revision_count} revision pass(es))")
        for gap in final.known_gaps:
            lines.append(f"- {gap}")
        lines.append("")

    lines.append(f"## Sources ({len(final.citations)})")
    for i, fact in enumerate(final.citations):
        lines.append(f"[{i + 1}] {fact.source_title} -- {fact.source_url}")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
