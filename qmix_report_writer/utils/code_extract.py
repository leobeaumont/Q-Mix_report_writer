"""Extract Python code from mixed text/code model output."""

import re


def extract_code(text: str) -> str:
    """Extract the best Python code from model output that may contain explanations.

    Tries multiple strategies in order:
    1. ```python ... ``` blocks
    2. ``` ... ``` blocks that look like Python
    3. Lines that look like Python code (def, class, import, etc.)
    """
    if not text or not text.strip():
        return ""

    # Strategy 1: explicit ```python blocks (take the longest one)
    python_blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if python_blocks:
        return max(python_blocks, key=len).strip()

    # Strategy 2: any ``` blocks that contain Python-like code
    any_blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    py_blocks = [b for b in any_blocks if _looks_like_python(b)]
    if py_blocks:
        return max(py_blocks, key=len).strip()

    # Strategy 3: if the whole text starts with a def/class/import, it's probably code
    stripped = text.strip()
    if re.match(r"^(def |class |import |from |#)", stripped):
        return stripped

    # Strategy 4: extract contiguous lines that look like code
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if _is_code_line(line):
            in_code = True
            code_lines.append(line)
        elif in_code and (line.strip() == "" or line.startswith(" ") or line.startswith("\t")):
            code_lines.append(line)
        elif in_code and not _is_code_line(line):
            if len(code_lines) > 3:
                break
            code_lines = []
            in_code = False

    if code_lines:
        return "\n".join(code_lines).strip()

    return stripped


def _looks_like_python(text: str) -> bool:
    indicators = ["def ", "class ", "import ", "return ", "for ", "while ", "if ", "print("]
    return any(ind in text for ind in indicators)


def _is_code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    starters = ["def ", "class ", "import ", "from ", "return ", "if ", "elif ",
                 "else:", "for ", "while ", "try:", "except", "with ", "raise ",
                 "yield ", "async ", "await ", "print(", "assert "]
    if any(stripped.startswith(s) for s in starters):
        return True
    if re.match(r"^\w+\s*[=\(]", stripped):
        return True
    if stripped.startswith("#"):
        return True
    return False
