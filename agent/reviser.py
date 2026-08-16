"""
Reviser stage.

Takes a draft report plus the Critic's flagged issues and produces a
corrected version. Runs in a loop with the Critic, capped at
Settings.max_revision_passes -- if issues remain after the cap, they
get surfaced honestly in the final report's known_gaps rather than
looped on forever or silently dropped.
"""

import anthropic

from .config import Settings
from .schemas import DraftReport, CritiqueResult
from .trace import Tracer

REVISER_SYSTEM_PROMPT = """You are the revision stage of a research agent. \
You are given a draft report and a list of specific issues a critic \
found in it. Produce a corrected version of the report.

Rules:
- Address every issue listed. For unsupported_claim issues, either \
remove the claim or find a citation number from the provided facts \
that actually supports it -- never invent a new citation number.
- For weak_sourcing issues, soften the claim's phrasing to match what \
a single non-authoritative source actually warrants (e.g. "one \
practitioner reported X" rather than "experienced engineers recommend X"), \
rather than removing the content entirely.
- For gap issues, add coverage using only the provided numbered facts \
-- if the facts genuinely don't cover it, say so explicitly in the \
report rather than fabricating content.
- For contradiction issues, acknowledge the conflict explicitly in the \
text rather than picking a side silently.
- Do not remove or weaken parts of the report that weren't flagged --  \
only change what the issues call out.
- Keep using [n] citation markers matching the same numbered facts \
list you were given.

Call the submit_report tool with the revised report. Do not respond \
with plain text.
"""

REVISED_REPORT_TOOL = {
    "name": "submit_report",
    "description": "Submit the revised research report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "body_markdown": {
                "type": "string",
                "description": "Full revised report body in markdown, with inline [n] citations",
            },
        },
        "required": ["title", "summary", "body_markdown"],
    },
}


class Reviser:
    def __init__(self, settings: Settings, tracer: Tracer):
        self.settings = settings
        self.tracer = tracer
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=settings.request_timeout_seconds)

    def revise(
        self,
        draft: DraftReport,
        critique: CritiqueResult,
        original_question: str,
    ) -> DraftReport:
        numbered_facts = "\n".join(
            f"[{i + 1}] {fact.fact} (source: {fact.source_title})"
            for i, fact in enumerate(draft.citations)
        )

        issues_text = "\n".join(
            f"- [{issue.severity}] {issue.location}: {issue.description}"
            for issue in critique.issues
        )

        response = self.client.messages.create(
            model=self.settings.model,
            max_tokens=8192,
            system=REVISER_SYSTEM_PROMPT,
            tools=[REVISED_REPORT_TOOL],
            tool_choice={"type": "tool", "name": "submit_report"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Original question: {original_question}\n\n"
                        f"Numbered facts:\n{numbered_facts}\n\n"
                        f"Current draft:\n\n{draft.title}\n\n{draft.summary}\n\n"
                        f"{draft.body_markdown}\n\n"
                        f"Issues to address:\n{issues_text}"
                    ),
                }
            ],
        )

        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                "Reviser response was cut off (hit max_tokens before finishing). "
                "Consider raising max_tokens further."
            )

        tool_call = _find_tool_use(response, "submit_report")
        parsed = tool_call.input

        missing_keys = [k for k in ("title", "summary", "body_markdown") if k not in parsed]
        if missing_keys:
            raise ValueError(
                f"Reviser tool call is missing required field(s): {missing_keys}."
            )

        revised = DraftReport(
            title=parsed["title"],
            summary=parsed["summary"],
            body_markdown=parsed["body_markdown"],
            citations=draft.citations,  # citation pool is unchanged by revision
        )

        self.tracer.log_stage(
            "reviser",
            {"question": original_question, "num_issues": len(critique.issues)},
            revised,
        )
        return revised


def _find_tool_use(response, tool_name: str):
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
            return block
    raise ValueError(
        f"Expected a '{tool_name}' tool call in the response but found none. "
        f"Block types present: {[getattr(b, 'type', type(b).__name__) for b in response.content]}"
    )