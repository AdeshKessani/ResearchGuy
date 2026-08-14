"""
Entry point.

Runs the full pipeline: Planner -> Researcher -> Synthesizer ->
Critic/Reviser loop -> final report. The revision loop is capped at
Settings.max_revision_passes -- if issues remain after the cap, they're
surfaced honestly in known_gaps rather than hidden or looped forever.
"""

import argparse
import time

from agent.config import load_settings
from agent.planner import Planner
from agent.researcher import Researcher
from agent.synthesizer import Synthesizer
from agent.critic import Critic
from agent.reviser import Reviser
from agent.schemas import FinalReport
from agent.trace import Tracer


def main():
    parser = argparse.ArgumentParser(description="Autonomous research agent")
    parser.add_argument("question", type=str, help="The research question")
    args = parser.parse_args()

    settings = load_settings()
    run_id = f"run_{int(time.time())}"
    tracer = Tracer(run_id)

    planner = Planner(settings, tracer)
    plan = planner.plan(args.question)
    print(f"\nOriginal question: {plan.original_question}\n")
    print("Sub-questions:")
    for sq in plan.sub_questions:
        print(f"  [{sq.id}] {sq.question}")
    print()

    researcher = Researcher(settings, tracer)
    print("Researching each sub-question (this calls Tavily + Haiku per source)...\n")
    results = researcher.research_all(plan.sub_questions)
    total_facts = sum(len(r.facts) for r in results)
    print(f"Total facts extracted: {total_facts} (capped at {settings.max_total_facts})\n")

    synthesizer = Synthesizer(settings, tracer)
    print("Synthesizing draft report...\n")
    draft = synthesizer.synthesize(plan.original_question, results)

    critic = Critic(settings, tracer)
    reviser = Reviser(settings, tracer)

    revision_count = 0
    critique = critic.critique(draft, plan.original_question)
    print(f"Critic pass 0: passes={critique.passes}, issues found={len(critique.issues)}")
    for issue in critique.issues:
        print(f"  [{issue.severity}] {issue.location}")
        print(f"      {issue.description}")

    while not critique.passes and revision_count < settings.max_revision_passes:
        revision_count += 1
        print(f"\nRevising (pass {revision_count})...\n")
        draft = reviser.revise(draft, critique, plan.original_question)
        critique = critic.critique(draft, plan.original_question)
        print(
            f"Critic pass {revision_count}: passes={critique.passes}, "
            f"issues found={len(critique.issues)}"
        )
        for issue in critique.issues:
            print(f"  [{issue.severity}] {issue.location}")
            print(f"      {issue.description}")

    final = FinalReport(
        title=draft.title,
        summary=draft.summary,
        body_markdown=draft.body_markdown,
        citations=draft.citations,
        known_gaps=[issue.description for issue in critique.issues],
        revision_count=revision_count,
    )

    print(f"\n=== {final.title} ===\n")
    print(final.summary)
    print()
    print(final.body_markdown)
    print()
    print(f"--- Citations ({len(final.citations)}) ---")
    for i, fact in enumerate(final.citations):
        print(f"[{i + 1}] {fact.fact}")
        print(f"    {fact.source_title} -- {fact.source_url}")

    if final.known_gaps:
        print(f"\n--- Known gaps (survived {revision_count} revision pass(es)) ---")
        for gap in final.known_gaps:
            print(f"  - {gap}")

    print(f"\n(trace written to traces/{run_id}.jsonl)")


if __name__ == "__main__":
    main()