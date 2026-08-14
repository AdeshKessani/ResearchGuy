"""
Synthesizer stage.

Takes every fact the Researcher collected (across all sub-questions) and
merges them into one coherent report answering the *original* question --
organized thematically, not by sub-question, with inline [n] citation
markers that map to real SourceFact objects.

Key design choice: facts are numbered *before* the LLM call, and the
model is instructed to only use those exact numbers as citation markers.
That numbering is what lets the Critic later verify every [n] in the
text maps to a real, correctly-supporting fact -- rather than trusting
the model's citations on faith.
"""

import anthropic

from .config import Settings
from .schemas import ResearchResult, SourceFact, DraftReport
from .trace import Tracer

SYNTHESIZER_SYSTEM_PROMPT = """You are the synthesis stage of a research agent.

You will be given a research question and a numbered list of facts \
gathered from web sources. Write a coherent report that answers the \
question.

Rules:
- Organize the report thematically -- group related facts together \
regardless of which sub-question they came from. Do not structure the \
report as a list of sub-question answers.
- Every factual claim in the body must cite its source using the exact \
fact number in brackets, e.g. "LangGraph provides built-in checkpointing [3]."
- Only use the fact numbers provided. Never invent a citation number \
that wasn't given to you.
- Do not state anything as fact that isn't backed by one of the \
provided numbered facts. If the provided facts don't fully answer some \
part of the question, say so explicitly in the report rather than \
filling the gap with unsupported claims.
- If facts conflict with each other, note the conflict explicitly in \
the text rather than silently picking one side.
- Write in clear, professional prose. Markdown formatting (headers, \
bullet points) is fine for structure.

Call the submit_report tool with your finished report. Do not respond \
with plain text.
"""

# Using tool_use instead of asking the model to hand-write JSON.
# body_markdown is long free text with quotes, headers, and nested
# punctuation -- exactly the content that breaks manually-parsed JSON
# when the model forgets to escape a character. Passing an input_schema
# and forcing tool_choice makes the API itself responsible for producing
# valid structured output, so this whole class of parse error goes away.
REPORT_TOOL = {
    "name": "submit_report",
    "description": "Submit the finished synthesized research report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Report title"},
            "summary": {
                "type": "string",
                "description": "2-3 sentence summary of the report's findings",
            },
            "body_markdown": {
                "type": "string",
                "description": "Full report body in markdown, with inline [n] citations",
            },
        },
        "required": ["title", "summary", "body_markdown"],
    },
}


class Synthesizer:
    def __init__(self, settings: Settings, tracer: Tracer):
        self.settings = settings
        self.tracer = tracer
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def synthesize(
        self, original_question: str, research_results: list[ResearchResult]
    ) -> DraftReport:
        all_facts: list[SourceFact] = [
            fact for result in research_results for fact in result.facts
        ]

        if not all_facts:
            # Nothing to synthesize -- surface this rather than asking
            # the model to write a report from an empty fact list.
            draft = DraftReport(
                title=original_question,
                summary="No facts were gathered for this question.",
                body_markdown=(
                    "No research facts were available to synthesize a report. "
                    "This likely means the Researcher stage failed to find or "
                    "extract relevant information for every sub-question."
                ),
                citations=[],
            )
            self.tracer.log_stage(
                "synthesizer", {"question": original_question, "num_facts": 0}, draft
            )
            return draft

        numbered_facts = "\n".join(
            f"[{i + 1}] {fact.fact} (source: {fact.source_title})"
            for i, fact in enumerate(all_facts)
        )

        response = self.client.messages.create(
            model=self.settings.model,
            # Raised from 3000 -- with 100+ facts to synthesize, the
            # report body alone can run long, and a cut-off tool call
            # was producing incomplete JSON (missing body_markdown)
            # rather than a clean error. 8192 gives real headroom.
            max_tokens=8192,
            system=SYNTHESIZER_SYSTEM_PROMPT,
            tools=[REPORT_TOOL],
            tool_choice={"type": "tool", "name": "submit_report"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Research question: {original_question}\n\n"
                        f"Numbered facts:\n{numbered_facts}"
                    ),
                }
            ],
        )

        if response.stop_reason == "max_tokens":
            # The model ran out of room mid-generation. Fail loudly and
            # specifically here rather than letting it surface as a
            # confusing KeyError three lines later.
            raise RuntimeError(
                "Synthesizer response was cut off (hit max_tokens before "
                "finishing). The report is likely too long for the current "
                "limit -- consider reducing sources_per_sub_question in "
                "Settings, or raising max_tokens further."
            )

        tool_call = _find_tool_use(response, "submit_report")
        parsed = tool_call.input  # already a validated dict, no JSON parsing needed

        missing_keys = [k for k in ("title", "summary", "body_markdown") if k not in parsed]
        if missing_keys:
            raise ValueError(
                f"Synthesizer tool call is missing required field(s): {missing_keys}. "
                f"stop_reason was '{response.stop_reason}'."
            )

        draft = DraftReport(
            title=parsed["title"],
            summary=parsed["summary"],
            body_markdown=parsed["body_markdown"],
            citations=all_facts,
        )

        self.tracer.log_stage(
            "synthesizer",
            {"question": original_question, "num_facts": len(all_facts)},
            draft,
        )
        return draft


def _find_tool_use(response, tool_name: str):
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
            return block
    raise ValueError(
        f"Expected a '{tool_name}' tool call in the response but found none. "
        f"Block types present: {[getattr(b, 'type', type(b).__name__) for b in response.content]}"
    )