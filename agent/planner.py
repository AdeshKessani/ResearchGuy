"""
Planner stage.

Takes a broad research question and decomposes it into a small set of
concrete, independently-searchable sub-questions. Quality here determines
the quality of everything downstream, so this gets its own tuning pass.
"""

import json
import anthropic

from .config import Settings
from .schemas import PlannerOutput, SubQuestion
from .trace import Tracer
from .llm_utils import extract_text

PLANNER_SYSTEM_PROMPT = """You are the planning stage of a research agent.

Given a broad research question, break it into 3-6 concrete sub-questions that:
- are each independently answerable via web search
- together cover the important dimensions of the original question
- avoid overlap with each other
- are specific enough to search well (not vague restatements of the original)

Respond with ONLY valid JSON matching this schema, no other text:
{
  "sub_questions": [
    {"id": "sq1", "question": "...", "rationale": "..."},
    ...
  ]
}
"""


class Planner:
    def __init__(self, settings: Settings, tracer: Tracer):
        self.settings = settings
        self.tracer = tracer
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def plan(self, question: str) -> PlannerOutput:
        response = self.client.messages.create(
            model=self.settings.model,
            max_tokens=1500,
            system=PLANNER_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Research question: {question}\n\n"
                        f"Produce at most {self.settings.max_sub_questions} sub-questions."
                    ),
                }
            ],
        )

        raw_text = extract_text(response)
        parsed = _parse_json(raw_text)

        sub_questions = [SubQuestion(**sq) for sq in parsed["sub_questions"]]
        sub_questions = sub_questions[: self.settings.max_sub_questions]

        output = PlannerOutput(original_question=question, sub_questions=sub_questions)

        self.tracer.log_stage("planner", question, output)
        return output


def _parse_json(text: str) -> dict:
    """Strip markdown code fences if the model added them, then parse."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())