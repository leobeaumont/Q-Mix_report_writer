"""Extract code blocks from LLM responses."""

import re


def extract_code(text: str, lang: str = "python") -> str:
    """Extract code from markdown code blocks or return raw text."""
    # ```python ... ```
    pattern = rf"```{re.escape(lang)}\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Generic ``` blocks
    matches = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return text.strip()
