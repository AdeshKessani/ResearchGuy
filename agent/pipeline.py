"""
Full pipeline runner: Planner -> Researcher -> Synthesizer ->
Critic/Reviser loop -> FinalReport.

Extracted from main.py so the same logic runs identically whether
invoked once from the CLI or across a whole eval test set -- a bug fix
here only needs to happen in one place.
"""

from .config import Settings
from .planner import Planner
from .researcher import Researcher
from .synthesizer import Synthesizer
from .critic import Critic
from .reviser import Reviser
from .schemas import FinalReport
from .trace import Tracer


def run_pipeline(
    question: str, settings: Settings, tracer: Tracer, verbose: bool = True
) -> FinalReport:
    def log(msg: str = ""):
        if verbose:
            print(msg)

    planner = Planner(settings, tracer)
    plan = planner.plan(question)
    log(f"\nOriginal question: {plan.original_question}\n")
    log("Sub-questions:")
    for sq in plan.sub_questions:
        log(f"  [{sq.id}] {sq.question}")
    log()

    researcher = Researcher(settings, tracer)
    log("Researching each sub-question (this calls Tavily + Haiku per source)...\n")
    results = researcher.research_all(plan.sub_questions)
    total_facts = sum(len(r.facts) for r in results)
    log(f"Total facts extracted: {total_facts} (capped at {settings.max_total_facts})\n")

    synthesizer = Synthesizer(settings, tracer)
    log("Synthesizing draft report...\n")
    draft = synthesizer.synthesize(plan.original_question, results)

    critic = Critic(settings, tracer)
    reviser = Reviser(settings, tracer)

    revision_count = 0
    critique = critic.critique(draft, plan.original_question)
    log(f"Critic pass 0: passes={critique.passes}, issues found={len(critique.issues)}")
    for issue in critique.issues:
        log(f"  [{issue.severity}] {issue.location}")
        log(f"      {issue.description}")

    while not critique.passes and revision_count < settings.max_revision_passes:
        revision_count += 1
        log(f"\nRevising (pass {revision_count})...\n")
        draft = reviser.revise(draft, critique, plan.original_question)
        critique = critic.critique(draft, plan.original_question)
        log(
            f"Critic pass {revision_count}: passes={critique.passes}, "
            f"issues found={len(critique.issues)}"
        )
        for issue in critique.issues:
            log(f"  [{issue.severity}] {issue.location}")
            log(f"      {issue.description}")

    return FinalReport(
        title=draft.title,
        summary=draft.summary,
        body_markdown=draft.body_markdown,
        citations=draft.citations,
        known_gaps=[issue.description for issue in critique.issues],
        revision_count=revision_count,
    )
