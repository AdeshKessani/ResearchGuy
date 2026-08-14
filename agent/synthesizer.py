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

import json
import anthropic

from .config import Settings
from .schemas import ResearchResult, SourceFact, DraftReport
from .trace import Tracer
from .llm_utils import extract_text

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

Respond with ONLY valid JSON, no other text:
{
  "title": "...",
  "summary": "2-3 sentence summary of the report's findings",
  "body_markdown": "full report body with inline [n] citations"
}
"""


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
            max_tokens=3000,
            system=SYNTHESIZER_SYSTEM_PROMPT,
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

        raw_text = extract_text(response)
        parsed = _parse_json(raw_text)

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


def _parse_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())