"""
Critic stage.

Checks the draft report against the *raw facts and their sources* --
not against the draft's own prose, which would just be checking a
thing against itself and would never catch anything. For every claim
in the report, the Critic can see exactly which fact and which source
title backs it, and is asked to judge both:

  1. Is this claim actually supported by the cited fact? (fabrication check)
  2. Is the citation itself trustworthy enough for the confidence with
     which the claim is stated? (source-quality check)

That second check exists because of a concrete finding from Phase 3: a
draft cited a single LinkedIn opinion post five times in a row ([99]-
[104]) to make one person's take read like an "experienced engineers
recommend" consensus. The report wasn't lying about what the source
said -- but it was overstating the authority of a single anecdote.
unsupported_claim doesn't catch that; weak_sourcing does.
"""

import json
import anthropic

from .config import Settings
from .schemas import DraftReport, SourceFact, CritiqueIssue, CritiqueResult
from .trace import Tracer

CRITIC_SYSTEM_PROMPT = """You are the critic stage of a research agent. \
You review a draft report against the numbered facts it was built from \
and flag problems before the report is finalized.

For each numbered fact you're given, you can see the fact text AND its \
source title -- use the source title to judge how authoritative it is \
(e.g. official documentation or a GitHub issue vs. a personal blog, \
forum comment, or LinkedIn post).

Check for four distinct kinds of issue:

1. unsupported_claim -- the report states something as fact that isn't \
actually backed by any of the cited numbered facts, or the cited fact \
doesn't really say what the report claims it says.

2. gap -- an important part of the original question isn't addressed \
by the report at all, or is addressed too thinly given what the facts \
actually support.

3. contradiction -- two or more facts conflict with each other, and the \
report doesn't acknowledge this, instead silently presenting one side.

4. weak_sourcing -- a claim is technically backed by a citation, and \
the citation does say what's claimed, BUT the claim is presented with \
more confidence or generality than a single non-authoritative source \
(personal opinion, forum comment, anecdote, marketing blog) warrants -- \
especially if that one source is cited repeatedly to make one person's \
view sound like a documented consensus.

For each issue found, give a specific location (quote or closely \
paraphrase the report text) and a clear description of the problem.

If the report has no real issues, return passes=true with an empty \
issues list. Do not invent issues to seem thorough -- a report that is \
genuinely solid should pass.

Call the submit_critique tool with your findings. Do not respond with \
plain text.
"""

CRITIQUE_TOOL = {
    "name": "submit_critique",
    "description": "Submit the critique findings for a draft report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "passes": {
                "type": "boolean",
                "description": "True if the report has no significant issues",
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": [
                                "unsupported_claim",
                                "gap",
                                "contradiction",
                                "weak_sourcing",
                            ],
                        },
                        "location": {
                            "type": "string",
                            "description": "Quote or paraphrase of the relevant report text",
                        },
                        "description": {"type": "string"},
                    },
                    "required": ["severity", "location", "description"],
                },
            },
        },
        "required": ["passes", "issues"],
    },
}


class Critic:
    def __init__(self, settings: Settings, tracer: Tracer):
        self.settings = settings
        self.tracer = tracer
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def critique(self, draft: DraftReport, original_question: str) -> CritiqueResult:
        numbered_facts = "\n".join(
            f"[{i + 1}] {fact.fact} (source: {fact.source_title})"
            for i, fact in enumerate(draft.citations)
        )

        response = self.client.messages.create(
            model=self.settings.model,
            # Raised from 4096 -- with 138 facts and a long draft to
            # review, the Critic can produce a long issues list, and a
            # cut-off tool call was producing incomplete JSON (missing
            # 'passes') instead of a clean error.
            max_tokens=8192,
            system=CRITIC_SYSTEM_PROMPT,
            tools=[CRITIQUE_TOOL],
            tool_choice={"type": "tool", "name": "submit_critique"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Original question: {original_question}\n\n"
                        f"Numbered facts used as sources:\n{numbered_facts}\n\n"
                        f"Draft report:\n\n{draft.title}\n\n{draft.summary}\n\n"
                        f"{draft.body_markdown}"
                    ),
                }
            ],
        )

        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                "Critic response was cut off (hit max_tokens before finishing). "
                "Consider raising max_tokens further."
            )

        tool_call = _find_tool_use(response, "submit_critique")
        parsed = tool_call.input

        raw_issues = _recover_issues_list(parsed.get("issues", []))
        issues = [_normalize_issue(raw) for raw in raw_issues]

        if "passes" in parsed:
            passes = parsed["passes"]
        else:
            # The model omitted 'passes' entirely even though it wasn't
            # truncated (stop_reason was 'tool_use', not 'max_tokens') --
            # this is model non-compliance with the schema, not a token
            # limit problem. Rather than crash on a field that's largely
            # redundant with the issues list anyway, derive it: no
            # issues found means it passes.
            passes = len(issues) == 0

        result = CritiqueResult(passes=passes, issues=issues)

        self.tracer.log_stage(
            "critic",
            {"question": original_question, "num_citations": len(draft.citations)},
            result,
        )
        return result


def _find_tool_use(response, tool_name: str):
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
            return block
    raise ValueError(
        f"Expected a '{tool_name}' tool call in the response but found none. "
        f"Block types present: {[getattr(b, 'type', type(b).__name__) for b in response.content]}"
    )


def _normalize_issue(raw) -> CritiqueIssue:
    """
    The tool schema asks for each issue as {severity, location, description},
    but tool_use schemas guide the model rather than strictly enforcing
    nested object structure -- occasionally an issue comes back as a
    plain string instead. Rather than crash the whole critique on one
    malformed entry, treat it as a valid issue with an 'unspecified'
    severity so it still surfaces to the user instead of being silently
    dropped or blowing up the pipeline.
    """
    if isinstance(raw, dict):
        return CritiqueIssue(
            severity=raw.get("severity", "unspecified"),
            location=raw.get("location", "unspecified"),
            description=raw.get("description", str(raw)),
        )
    # Fallback: the model returned a bare string for this issue.
    return CritiqueIssue(
        severity="unspecified",
        location="unspecified",
        description=str(raw),
    )


def _recover_issues_list(raw_issues) -> list:
    """
    Guards against a real failure mode observed in practice: the model
    sometimes nests a full stringified copy of its own JSON output
    inside the 'issues' field as a string, instead of a real array.
    Without this check, `for raw in parsed["issues"]` silently iterates
    a string character-by-character (Python strings are iterable),
    producing hundreds of single-letter "issues" instead of crashing --
    which is worse than a crash, because it fails silently with
    plausible-looking output.

    If issues is already a proper list, pass it through unchanged. If
    it's a string, try to recover the real list by parsing it as JSON
    (it may itself be an issues array, or a full {"passes":.., "issues":
    [...]} object). If recovery isn't possible, return an empty list
    rather than iterating the string.
    """
    if isinstance(raw_issues, list):
        return raw_issues

    if isinstance(raw_issues, str):
        try:
            recovered = json.loads(raw_issues)
        except json.JSONDecodeError:
            return []
        if isinstance(recovered, list):
            return recovered
        if isinstance(recovered, dict) and isinstance(recovered.get("issues"), list):
            return recovered["issues"]
        return []

    return []