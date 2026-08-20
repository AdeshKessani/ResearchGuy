"""
Code-based citation integrity checks.

Deliberately not an LLM call -- whether every [n] marker in the report
body maps to a real citation index is a mechanical fact, not a judgment
call, so it should be checked mechanically. Keeping this separate from
the LLM-as-judge in judge.py also means this metric can never be wrong
because a judge model had a bad day.
"""

import re

from agent.schemas import FinalReport


def citation_integrity(report: FinalReport) -> dict:
    body = report.body_markdown
    num_citations = len(report.citations)

    all_bracket_numbers = [int(n) for n in re.findall(r"\[(\d+)\]", body)]

    # A source's own title can contain a bracketed year -- e.g. a blog
    # titled "...When NoSQL Actually Wins [2026]" -- and the Synthesizer
    # faithfully quotes that title into the report body. A naive regex
    # can't tell that "[2026]" from a real citation marker like "[47]",
    # and with max_total_facts capped at 130, no genuine citation number
    # will ever land in a year-like range anyway. Excluded here rather
    # than silently dropped, so a genuinely huge citation count wouldn't
    # get miscategorized without a trace.
    def looks_like_year(n: int) -> bool:
        return 1900 <= n <= 2099

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
