# Autonomous Research Agent

An agent that takes a broad research question, decomposes it into
sub-questions, researches each one on the web with citations, drafts a
report, then critiques and revises its own draft before finalizing
with an independent evaluation harness measuring how much slips through
even after self-correction.

## Architecture

```
question -> Planner -> [sub-questions]
                          -> Researcher (per sub-question, web search + fact extraction)
                              -> Synthesizer -> draft report
                                  -> Critic -> issues found?
                                       -> Reviser -> final report
                                            (capped at 2 revision passes)
```

Every stage passes a validated, typed object to the next (see
`agent/schemas.py`) and logs its input/output to `traces/<run_id>.jsonl`,
so the full reasoning chain for any run can be replayed and inspected.

**Stack:** Python, Claude Sonnet 5 (reasoning-heavy stages) + Claude
Haiku 4.5 (high-volume fact extraction), Tavily (web search), Pydantic
(structured data), Claude's tool-use feature (structured API output).

## What each stage does

- **Planner:** breaks the question into 3-6 independently-searchable
  sub-questions.
- **Researcher:** searches the web per sub-question (Tavily), extracts
  atomic facts with source URLs attached (Haiku). Total facts capped at
  130 to keep downstream stages' input bounded.
- **Synthesizer:** merges facts into one report, organized thematically
  rather than by sub-question, with inline `[n]` citations tied to real
  source facts.
- **Critic:** checks the draft against the raw facts for four issue types: unsupported claims, gaps, contradictions, and weak sourcing.
- **Reviser:** addresses flagged issues; for weak-sourcing issues,
  softens the claim's phrasing rather than deleting the content.
- **Eval harness:** (`eval/`) runs the pipeline against a held-out
  question set and scores each result two ways: a deterministic
  citation-integrity check (no LLM involved), and an independent
  LLM judge that reviews the *final* report, after the internal
  Critic/Reviser loop has already run, to measure what actually
  survives self-correction.

## Pilot evaluation results

Run against a 2-question pilot set (one technical, one comparative) due
to API cost constraints during development. Full results in
`eval_results.json`.

| Question | Category | Revisions | Citation integrity | Judge passes | Judge issues |
|---|---|---|---|---|---|
| LangGraph vs. hand-rolled agent loop | technical | 2 | pass (108/114 citations used, 0 out of range) | pass | 4 (all minor -- inferential reasoning presented within factual narrative) |
| PostgreSQL vs. MongoDB for high-write analytics | comparative | 1 | pass (127/130 citations used, 0 out of range) | fail | 3 (one genuine unsupported technical claim; two minor) |

**Honest takeaways from this pilot:**
- Citation integrity was clean on both run, no fabrication or hallucination.
- The in-loop Critic and the independent post-hoc Judge catch
  *different* things. The Critic (which runs before the Reviser fixes
  anything) caught and resolved several weak-sourcing issues on both
  runs. The Judge, checking the *final* report afterward, still found
  the model occasionally inserting its own unstated reasoning into the
  narrative (e.g. speculatively reconciling two conflicting
  benchmarks).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and TAVILY_API_KEY
```

## Usage

Single question:
```bash
python main.py "What are the tradeoffs between building agents with LangGraph vs a hand-rolled loop?"
```

Run the eval harness:
```bash
python -m eval.run_eval
```
