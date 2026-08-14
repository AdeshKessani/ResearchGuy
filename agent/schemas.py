"""
Shared data models for the research agent pipeline.

Every stage (Planner -> Researcher -> Synthesizer -> Critic -> Reviser)
passes structured Pydantic objects to the next, rather than raw strings.
This is what lets us log/trace each stage and validate outputs.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    id: str = Field(..., description="Short id, e.g. 'sq1'")
    question: str = Field(..., description="The sub-question text")
    rationale: str = Field(
        ..., description="Why this sub-question matters for the parent question"
    )


class PlannerOutput(BaseModel):
    original_question: str
    sub_questions: list[SubQuestion]


class SourceFact(BaseModel):
    sub_question_id: str
    fact: str = Field(..., description="A single extracted, atomic claim")
    source_url: str
    source_title: str


class ResearchResult(BaseModel):
    sub_question_id: str
    query_used: str
    facts: list[SourceFact]
    notes: Optional[str] = Field(
        default=None, description="Any caveats, e.g. 'sources conflicted on X'"
    )


class DraftReport(BaseModel):
    title: str
    summary: str
    body_markdown: str = Field(
        ..., description="Full report body with inline [n] citation markers"
    )
    citations: list[SourceFact]


class CritiqueIssue(BaseModel):
    severity: str = Field(
        ...,
        description=(
            "'unsupported_claim' | 'gap' | 'contradiction' | 'weak_sourcing'. "
            "weak_sourcing flags claims presented with unwarranted confidence "
            "when they rest on a single non-authoritative source (e.g. an "
            "opinion post or forum comment) -- distinct from unsupported_claim, "
            "since the claim IS backed by a citation, just a weak one."
        ),
    )
    location: str = Field(..., description="Which part of the report this refers to")
    description: str


class CritiqueResult(BaseModel):
    passes: bool
    issues: list[CritiqueIssue]


class FinalReport(BaseModel):
    title: str
    summary: str
    body_markdown: str
    citations: list[SourceFact]
    known_gaps: list[str] = Field(
        default_factory=list,
        description="Honest gaps that survived revision passes",
    )
    revision_count: int
