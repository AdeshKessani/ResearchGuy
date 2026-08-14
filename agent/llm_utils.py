"""
Small shared utility for calling the Anthropic API across all stages.
"""


def extract_text(response) -> str:
    """
    Pull the text content out of an Anthropic API response.

    response.content is a list of blocks that can include ThinkingBlock,
    ToolUseBlock, etc. depending on model and settings -- text is not
    guaranteed to be at index 0 (e.g. extended thinking puts a
    ThinkingBlock first). This finds the first actual text block instead
    of assuming position, which is what caused the AttributeError.
    """
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError(
        f"No text block found in API response. Block types present: "
        f"{[getattr(b, 'type', type(b).__name__) for b in response.content]}"
    )