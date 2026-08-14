"""
Researcher stage.

For each sub-question from the Planner:
  1. Search the web (Tavily) -- no LLM cost, just an API call
  2. For each of the top N results, extract atomic facts via Haiku
     (cheap, high-volume, narrow extraction task -- not open-ended
     reasoning, so a smaller model is the right tool here)

Every fact keeps its source_url and source_title attached at extraction
time. This is the step that makes citations possible later -- if we
extracted paragraphs instead of atomic, source-tagged facts, the
Synthesizer and Critic downstream would have nothing precise to check.
"""

import json
import anthropic
from tavily import TavilyClient

from .config import Settings
from .schemas import SubQuestion, SourceFact, ResearchResult
from .trace import Tracer
from .llm_utils import extract_text

EXTRACTION_SYSTEM_PROMPT = """You extract atomic facts from a piece of text, \
relevant to a specific research question.

Rules:
- Each fact must be a single, standalone, checkable claim
- Only extract facts that are actually relevant to the question
- Do not infer or add anything not stated in the text
- If the text has nothing relevant, return an empty list
- Skip marketing language, opinions, and vague claims

Respond with ONLY valid JSON, no other text:
{"facts": ["fact one", "fact two", ...]}
"""


class Researcher:
    def __init__(self, settings: Settings, tracer: Tracer):
        self.settings = settings
        self.tracer = tracer
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.tavily = TavilyClient(api_key=settings.tavily_api_key)

    def research(self, sub_question: SubQuestion) -> ResearchResult:
        search_response = self.tavily.search(
            query=sub_question.question,
            max_results=self.settings.sources_per_sub_question,
        )
        results = search_response.get("results", [])

        all_facts: list[SourceFact] = []
        notes = None

        if not results:
            notes = f"No search results found for: {sub_question.question}"
        else:
            for result in results:
                content = result.get("content", "")
                url = result.get("url", "")
                title = result.get("title", url)

                if not content:
                    continue

                extracted = self._extract_facts(sub_question.question, content)
                for fact_text in extracted:
                    all_facts.append(
                        SourceFact(
                            sub_question_id=sub_question.id,
                            fact=fact_text,
                            source_url=url,
                            source_title=title,
                        )
                    )

            if not all_facts:
                notes = (
                    f"Search returned {len(results)} results but no relevant "
                    f"facts were extracted -- sources may not have addressed "
                    f"the question directly."
                )

        result = ResearchResult(
            sub_question_id=sub_question.id,
            query_used=sub_question.question,
            facts=all_facts,
            notes=notes,
        )

        self.tracer.log_stage(
            f"researcher:{sub_question.id}",
            {"question": sub_question.question, "num_sources": len(results)},
            result,
        )
        return result

    def research_all(self, sub_questions: list[SubQuestion]) -> list[ResearchResult]:
        results = [self.research(sq) for sq in sub_questions]
        return _cap_total_facts(results, self.settings.max_total_facts)

    def _extract_facts(self, question: str, content: str) -> list[str]:
        # Truncate long pages -- extraction doesn't need the whole article,
        # and this keeps Haiku calls cheap and fast.
        truncated = content[:4000]

        response = self.client.messages.create(
            model=self.settings.researcher_model,
            max_tokens=500,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nText:\n{truncated}",
                }
            ],
        )

        raw_text = extract_text(response)
        try:
            parsed = _parse_json(raw_text)
            return parsed.get("facts", [])
        except (json.JSONDecodeError, IndexError):
            # A malformed extraction shouldn't crash the whole run --
            # skip this source's facts and let the pipeline continue.
            return []


def _parse_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def _cap_total_facts(
    results: list[ResearchResult], max_total: int
) -> list[ResearchResult]:
    """
    Trim the total fact count across all sub-questions down to max_total,
    removing evenly from whichever sub-question currently has the most
    facts rather than just truncating the tail of the list. This avoids
    one sub-question (e.g. the one researched first) silently keeping
    all its facts while a later one gets starved.
    """
    total = sum(len(r.facts) for r in results)
    if total <= max_total:
        return results

    # Work on plain lists we can mutate, keyed by sub_question_id.
    facts_by_id = {r.sub_question_id: list(r.facts) for r in results}
    overflow = total - max_total

    for _ in range(overflow):
        # Remove one fact from whichever sub-question currently has the
        # most -- keeps trimming balanced across sub-questions.
        fattest_id = max(facts_by_id, key=lambda k: len(facts_by_id[k]))
        if facts_by_id[fattest_id]:
            facts_by_id[fattest_id].pop()

    return [
        ResearchResult(
            sub_question_id=r.sub_question_id,
            query_used=r.query_used,
            facts=facts_by_id[r.sub_question_id],
            notes=r.notes,
        )
        for r in results
    ]