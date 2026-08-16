"""
LLM-as-judge for the eval harness.

This is intentionally separate from agent/critic.py, even though the
job sounds similar. The Critic runs *inside* the pipeline and its
findings get fixed by the Reviser before the run ever finishes -- so by
definition, whatever the Critic catches doesn't survive into the final
report. The Judge's job is different: it checks the *final* report,
after self-correction has already happened, to measure how much
actually slips through. That residual rate is the real reliability
number for the resume writeup -- "the agent catches its own mistakes"
is only provable by an independent check downstream of the fix.

Built with the same defensive patterns Phase 4 needed in practice:
tool-use for structured output, passes derived from findings when the
model omits it, and type-checked list recovery.
"""

import json
import anthropic

from agent.config import Settings
from agent.schemas import FinalReport

JUDGE_SYSTEM_PROMPT = """You are an independent evaluator reviewing a \
finished research report for reliability -- NOT part of the system \
that produced it. Assume the report has already been through an \
internal self-review process; your job is to catch anything that \
still slipped through.

You will be given the original question, the numbered facts the report \
was built from (with source titles), and the finished report.

Check for:
- unsupported_claim: a claim in the report not actually backed by any \
of the numbered facts, or a citation that doesn't say what's claimed
- weak_sourcing: a claim stated with more confidence than a single \
non-authoritative source (opinion post, forum comment, marketing blog) \
warrants
- contradiction: facts that conflict, silently resolved one way without \
acknowledgment

Call the submit_judgment tool with your findings. Do not respond with \
plain text.
"""

JUDGE_TOOL = {
    "name": "submit_judgment",
    "description": "Submit the independent evaluation of a finished report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "passes": {
                "type": "boolean",
                "description": "True if no significant issues remain",
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["unsupported_claim", "weak_sourcing", "contradiction"],
                        },
                        "description": {"type": "string"},
                    },
                    "required": ["severity", "description"],
                },
            },
        },
        "required": ["passes", "issues"],
    },
}


class Judge:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=settings.request_timeout_seconds)

    def judge(self, report: FinalReport, original_question: str) -> dict:
        numbered_facts = "\n".join(
            f"[{i + 1}] {fact.fact} (source: {fact.source_title})"
            for i, fact in enumerate(report.citations)
        )

        response = self.client.messages.create(
            model=self.settings.model,
            max_tokens=4096,
            system=JUDGE_SYSTEM_PROMPT,
            tools=[JUDGE_TOOL],
            tool_choice={"type": "tool", "name": "submit_judgment"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Original question: {original_question}\n\n"
                        f"Numbered facts:\n{numbered_facts}\n\n"
                        f"Final report:\n\n{report.title}\n\n{report.summary}\n\n"
                        f"{report.body_markdown}"
                    ),
                }
            ],
        )

        if response.stop_reason == "max_tokens":
            return {
                "passes": False,
                "issues": [
                    {
                        "severity": "eval_error",
                        "description": "Judge response was truncated (max_tokens)",
                    }
                ],
            }

        tool_call = _find_tool_use(response, "submit_judgment")
        parsed = tool_call.input

        raw_issues = _recover_issues_list(parsed.get("issues", []))
        issues = [_normalize_issue(raw) for raw in raw_issues]

        passes = parsed["passes"] if "passes" in parsed else (len(issues) == 0)

        return {"passes": passes, "issues": issues}


def _find_tool_use(response, tool_name: str):
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
            return block
    raise ValueError(
        f"Expected a '{tool_name}' tool call but found none. "
        f"Block types present: {[getattr(b, 'type', type(b).__name__) for b in response.content]}"
    )


def _normalize_issue(raw) -> dict:
    if isinstance(raw, dict):
        return {
            "severity": raw.get("severity", "unspecified"),
            "description": raw.get("description", str(raw)),
        }
    return {"severity": "unspecified", "description": str(raw)}


def _recover_issues_list(raw_issues) -> list:
    """Same recovery logic as agent/critic.py -- see that module's
    docstring for the concrete failure mode this guards against."""
    if isinstance(raw_issues, list):
        return raw_issues
    if isinstance(raw_issues, str):
        try:
            recovered = json.loads(raw_issues)
        except json.JSONDecodeError:
            return []
        if isinstance(recovered, list):
            return recovered
        if isinstance(recovered, dict) and isinstance(recovered.get("issues"), list):
            return recovered["issues"]
        return []
    return []