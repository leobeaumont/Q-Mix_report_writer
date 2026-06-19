import re
import json
import hashlib
from typing import Any, Dict, List, Optional

def safe_json_parse(text):
    """Clean markdown and attempt to fix truncated JSON."""
    if not text:
        return {}

    # 1. Strip markdown code fences
    text = re.sub(r"```json\s*|\s*```", "", text).strip()

    # 2. Direct parse (happy path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Model added preamble/postamble — find the first { ... } block
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # 4. Last resort: try to close an unterminated object
    if text.count('"') % 2 != 0:
        text += '"'
    if not text.endswith("}"):
        text += "}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"CRITICAL: Failed to parse LLM response: {text[:100]}...")
        return {}

def extract_number(text: str) -> Optional[float]:
    """Extract the last number from text (for math answers)."""
    matches = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return float(matches[-1]) if matches else None


def extract_code_block(text: str, lang: str = "python") -> str:
    """Extract code from markdown code blocks."""
    pattern = rf"```{lang}\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_choice(text: str, choices: List[str] = None) -> str:
    """Extract a multiple choice answer (A/B/C/D) from text."""
    text_upper = text.strip().upper()
    match = re.search(r"\b([A-D])\b", text_upper)
    if match:
        return match.group(1)
    for letter in ["A", "B", "C", "D"]:
        if letter in text_upper:
            return letter
    return text.strip()[:1].upper()


def hash_task(task: str) -> str:
    """Deterministic hash for caching."""
    return hashlib.md5(task.encode()).hexdigest()[:12]


def save_jsonl(path: str, data: List[Dict]):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results
