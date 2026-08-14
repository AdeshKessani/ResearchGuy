"""
Entry point.

Currently runs Planner -> Researcher -> Synthesizer end to end. Check
that every [n] citation in the printed report actually corresponds to
a real fact below it, and that the report doesn't state anything the
facts don't support. Critic/Reviser get wired in next.
"""

import argparse
import time

from agent.config import load_settings
from agent.planner import Planner
from agent.researcher import Researcher
from agent.synthesizer import Synthesizer
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
    print(f"Total facts extracted: {total_facts}\n")

    synthesizer = Synthesizer(settings, tracer)
    print("Synthesizing draft report...\n")
    draft = synthesizer.synthesize(plan.original_question, results)

    print(f"=== {draft.title} ===\n")
    print(draft.summary)
    print()
    print(draft.body_markdown)
    print()
    print(f"--- Citations ({len(draft.citations)}) ---")
    for i, fact in enumerate(draft.citations):
        print(f"[{i + 1}] {fact.fact}")
        print(f"    {fact.source_title} -- {fact.source_url}")

    print(f"\n(trace written to traces/{run_id}.jsonl)")


if __name__ == "__main__":
    main()