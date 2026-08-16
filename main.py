"""
Entry point.

Runs the full pipeline (see agent/pipeline.py) for a single question
from the command line and prints the final report.
"""

import argparse
import time

from agent.config import load_settings
from agent.pipeline import run_pipeline
from agent.trace import Tracer


def main():
    parser = argparse.ArgumentParser(description="Autonomous research agent")
    parser.add_argument("question", type=str, help="The research question")
    args = parser.parse_args()

    settings = load_settings()
    run_id = f"run_{int(time.time())}"
    tracer = Tracer(run_id)

    final = run_pipeline(args.question, settings, tracer, verbose=True)

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
        print(f"\n--- Known gaps (survived {final.revision_count} revision pass(es)) ---")
        for gap in final.known_gaps:
            print(f"  - {gap}")

    print(f"\n(trace written to traces/{run_id}.jsonl)")


if __name__ == "__main__":
    main()