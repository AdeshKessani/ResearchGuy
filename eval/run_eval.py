"""
Eval harness entry point.

Runs the full pipeline against every question in eval/dataset.py,
scores each result with the code-based citation check and the
independent LLM judge, and prints a results table -- plus writes
eval_results.json so you have a concrete artifact for the resume
writeup, not just console output that scrolls away.

Usage:
    python -m eval.run_eval
"""

import json
import time

from agent.config import load_settings
from agent.pipeline import run_pipeline
from agent.trace import Tracer
from eval.dataset import EVAL_DATASET
from eval.citation_metrics import citation_integrity
from eval.judge import Judge


def main():
    settings = load_settings()
    judge = Judge(settings)

    results = []

    for i, item in enumerate(EVAL_DATASET):
        print(f"\n[{i + 1}/{len(EVAL_DATASET)}] ({item.category}) {item.question}")

        run_id = f"eval_{int(time.time())}_{i}"
        tracer = Tracer(run_id)

        try:
            report = run_pipeline(item.question, settings, tracer, verbose=False)
        except Exception as e:
            print(f"  PIPELINE ERROR: {e}")
            results.append(
                {
                    "question": item.question,
                    "category": item.category,
                    "error": str(e),
                }
            )
            continue

        citation_metrics = citation_integrity(report)
        judgment = judge.judge(report, item.question)

        result = {
            "question": item.question,
            "category": item.category,
            "revision_count": report.revision_count,
            "known_gaps_count": len(report.known_gaps),
            "citation_integrity": citation_metrics,
            "judge_passes": judgment["passes"],
            "judge_issues": judgment["issues"],
            "run_id": run_id,
        }
        results.append(result)

        status = "PASS" if (citation_metrics["citation_integrity_passed"] and judgment["passes"]) else "FLAGGED"
        print(f"  [{status}] revisions={report.revision_count}, "
              f"citation_integrity={citation_metrics['citation_integrity_passed']}, "
              f"judge_passes={judgment['passes']}, "
              f"judge_issues={len(judgment['issues'])}")

    _print_summary(results)

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results written to eval_results.json")


def _print_summary(results: list[dict]):
    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)

    completed = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]

    print(f"\n{len(completed)}/{len(results)} questions completed without pipeline errors")
    if errored:
        print(f"{len(errored)} pipeline errors:")
        for r in errored:
            print(f"  - {r['question'][:60]}: {r['error'][:100]}")

    if not completed:
        return

    citation_pass_count = sum(
        1 for r in completed if r["citation_integrity"]["citation_integrity_passed"]
    )
    judge_pass_count = sum(1 for r in completed if r["judge_passes"])
    avg_revisions = sum(r["revision_count"] for r in completed) / len(completed)
    avg_usage_rate = sum(
        r["citation_integrity"]["citation_usage_rate"] for r in completed
    ) / len(completed)

    print(f"\nCitation integrity: {citation_pass_count}/{len(completed)} "
          f"({100 * citation_pass_count / len(completed):.0f}%) "
          f"had zero out-of-range citations")
    print(f"Judge pass rate:    {judge_pass_count}/{len(completed)} "
          f"({100 * judge_pass_count / len(completed):.0f}%) "
          f"had no unsupported claims / weak sourcing / contradictions post-revision")
    print(f"Avg revision passes: {avg_revisions:.1f}")
    print(f"Avg citation usage rate: {avg_usage_rate:.1%}")

    print(f"\n{'Category':<15} {'Question':<45} {'Cite OK':<9} {'Judge':<7} {'Revs':<5}")
    print("-" * 85)
    for r in completed:
        cite_ok = "yes" if r["citation_integrity"]["citation_integrity_passed"] else "NO"
        judge_ok = "yes" if r["judge_passes"] else "NO"
        print(
            f"{r['category']:<15} {r['question'][:43]:<45} "
            f"{cite_ok:<9} {judge_ok:<7} {r['revision_count']:<5}"
        )


if __name__ == "__main__":
    main()
