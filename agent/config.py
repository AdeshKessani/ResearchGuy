"""
Loads configuration from environment variables (via .env in development).
"""

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional; in prod you'd set real env vars instead
    pass


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    tavily_api_key: str
    # Reasoning-heavy stages (decomposition, synthesis, critique) get Sonnet.
    model: str = "claude-sonnet-5"
    # The Researcher stage makes many more calls than any other stage
    # (up to sources_per_sub_question x max_sub_questions per run), and
    # each call is a narrow extraction task, not open-ended reasoning.
    # Haiku is meaningfully cheaper here with no real quality loss for
    # "pull the facts out of this text" work.
    researcher_model: str = "claude-haiku-4-5-20251001"
    max_sub_questions: int = 6
    sources_per_sub_question: int = 3
    max_revision_passes: int = 2


def load_settings() -> Settings:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")

    missing = [
        name
        for name, val in [
            ("ANTHROPIC_API_KEY", anthropic_key),
            ("TAVILY_API_KEY", tavily_key),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill in your keys."
        )

    return Settings(anthropic_api_key=anthropic_key, tavily_api_key=tavily_key)