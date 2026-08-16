"""
Code-based citation integrity checks.

Deliberately not an LLM call. Whether every [n] marker in the report
body maps to a real citation index is a mechanical fact, not a judgment
call, so it should be checked mechanically.
"""

import re

from agent.schemas import FinalReport


def citation_integrity(report: FinalReport) -> dict:
    body = report.body_markdown
    num_citations = len(report.citations)

    all_bracket_numbers = [int(n) for n in re.findall(r"\[(\d+)\]", body)]

    # So thqat the metric doesn't get confused by years in the text, ignore any
    # bracketed numbers that look like years (1900-2099). This is a
    # heuristic, but it should be good enough for this purpose.
    def looks_like_year(n: int) -> bool:
        return 1800 <= n <= 2099

    citation_numbers = [n for n in all_bracket_numbers if not looks_like_year(n)]
    excluded_as_year = sorted(set(n for n in all_bracket_numbers if looks_like_year(n)))

    used_numbers = sorted(set(citation_numbers))
    out_of_range = [n for n in used_numbers if n < 1 or n > num_citations]

    usage_rate = (len(used_numbers) / num_citations) if num_citations else 0.0

    return {
        "num_citations_available": num_citations,
        "num_citation_markers_used": len(used_numbers),
        "out_of_range_citations": out_of_range,
        "citation_integrity_passed": len(out_of_range) == 0,
        "citation_usage_rate": round(usage_rate, 3),
        "excluded_as_likely_year": excluded_as_year,
    }