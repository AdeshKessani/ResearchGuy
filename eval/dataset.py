"""
Eval test set.

Spans the same domain categories used for Phase 1's Planner acceptance
check (technical, current-events, comparative, historical), so this
dataset does double duty: it's what you already used to judge the
Planner by hand, and now it's what the automated eval runs against.
"""

from dataclasses import dataclass


@dataclass
class EvalQuestion:
    question: str
    category: str


EVAL_DATASET: list[EvalQuestion] = [
    EvalQuestion(
        "What are the tradeoffs between building agents with LangGraph vs a hand-rolled loop?",
        "technical",
    ),
    EvalQuestion(
        "PostgreSQL vs MongoDB for a high-write analytics workload -- which is better and why?",
        "comparative",
    ),
]