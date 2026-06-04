from typing import List, Optional, Dict
import json

class Singleton:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self):
        self.value = 0.0


class Cost(Singleton):
    def __init__(self):
        self.value = 0.0


class PromptTokens(Singleton):
    def __init__(self):
        self.value = 0.0


class CompletionTokens(Singleton):
    def __init__(self):
        self.value = 0.0


class Time(Singleton):
    def __init__(self):
        self.value = ""


class Mode(Singleton):
    def __init__(self):
        self.value = ""

class ReportState(Singleton):
    def __init__(self):
        self.content = ""
        self.sections: List[Dict] = []   # structured view for in-place editing
        self.additions: List[str] = []
        self.sources: List = []
        self.progress = "[NOTHING WRITTEN SO FAR]"
        self.task = "[DO NOT PROCEED, WAIT FOR LEAD ARCHITECT TO ASSIGN A TASK]"
        self.deficient_topics: List[str] = []  # topics absent from the knowledge base
        self.review_section_idx: int = 0  # current section index during SECTION_REVIEW

    def reset(self):
        self.content = ""
        self.sections = []
        self.additions = []
        self.sources = []
        self.progress = "[NOTHING WRITTEN SO FAR]"
        self.task = "[DO NOT PROCEED, WAIT FOR LEAD ARCHITECT TO ASSIGN A TASK]"
        self.deficient_topics = []
        self.review_section_idx = 0

    def add_deficiency(self, topic: str) -> None:
        """Record a topic the Researcher confirmed is absent from the knowledge base."""
        topic = topic.strip()
        if topic and topic not in self.deficient_topics:
            self.deficient_topics.append(topic)

    def append(self, text: str, progress: str, new_sources: Optional[List] = None):
        """Append a new section. Infers title from first Markdown heading."""
        first_line = text.strip().split("\n")[0] if text.strip() else ""
        title = first_line.lstrip("#").strip() if first_line.startswith("#") else ""
        section_id = f"section_{len(self.sections) + 1}"
        self.sections.append({"id": section_id, "title": title, "content": text})
        self.content = "\n\n".join(s["content"] for s in self.sections)
        self.additions.append(text)
        self.progress = progress
        if new_sources is not None:
            self.sources += new_sources
        return section_id

    def replace_section(self, section_id: str, new_content: str) -> bool:
        """Replace an existing section's content in-place.

        Also rebuilds self.content so all existing readers stay correct.
        Returns False if section_id is not found.
        """
        for section in self.sections:
            if section["id"] == section_id:
                first_line = new_content.strip().split("\n")[0] if new_content.strip() else ""
                if first_line.startswith("#"):
                    section["title"] = first_line.lstrip("#").strip()
                section["content"] = new_content
                self.content = "\n\n".join(s["content"] for s in self.sections)
                return True
        return False

    def list_sections(self, verbose: bool = False) -> str:
        """Formatted section index for agent context (REVIEW / REVISION phases).

        verbose=True appends a one-line content excerpt per section so agents
        can identify sections by content rather than title alone, preventing
        ID mismatches when two sections share similar titles.
        """
        if not self.sections:
            return "[No sections written yet]"
        lines = []
        for s in self.sections:
            entry = f"- {s['id']}: {s['title'] or '(untitled)'}"
            if verbose:
                body = [
                    l for l in s["content"].strip().splitlines()
                    if l.strip() and not l.lstrip().startswith("#")
                ]
                if body:
                    excerpt = " ".join(body[0].split())[:120]
                    entry += f"\n  ↳ {excerpt}…"
            lines.append(entry)
        return "\n".join(lines)

    def get_last(self) -> str:
        if self.sections:
            return self.sections[-1]["content"]
        return "[NO PREVIOUS TEXT]"
    
class Score(Singleton):
    def __init__(self):
        self.previous_score: Optional[float] = None
        self.current_score: Optional[float] = None
        self.micro_scores: List[float] = []
        self.micro_notes: List[str] = []

    def reset(self):
        self.previous_score = None
        self.current_score = None
        self.micro_scores = []
        self.micro_notes = []

    def get_delta(self):
        if self.previous_score is None:
            return self.current_score
        else:
            return self.current_score - self.previous_score
        
    def update(self, new_score):
        self.previous_score = self.current_score
        self.current_score = new_score

class LengthGoal(Singleton):
    def __init__(self):
        self.previous_score: Optional[float] = None
        self.current_score: Optional[float] = None

    def reset(self):
        self.previous_score = None
        self.current_score = None

    def get_delta(self):
        if self.previous_score is None:
            return self.current_score
        else:
            return self.current_score - self.previous_score
        
    def update(self, new_score):
        self.previous_score = self.current_score
        self.current_score = new_score

class SourceBuffer(Singleton):
    def __init__(self):
        self.sources = []

    def reset(self):
        self.sources = []

    def add(self, source: Dict):
        self.sources.append(source)

    def flush(self) -> List[Dict]:
        out = self.sources
        self.reset()
        return out

class ExecutionTrace(Singleton):
    """ 
    Log of the whole execution process.

    The trace is a list of round, the data of each round is stored in a dict:

    trace = [
        # Round 1 data
        {
            "exec_order": [0, 1, 2, 3, 4, 5, 6]  # Agent exec order
            "agent_0": {
                "action": 0
                "message_to": [4, 5]
                "prompt": agent0.prompt
                "response": agent0.response
            },
            "agent_1": {

                ...

            },

            ...

            "agent_5": {
            
                ...
            
            }
        }
    ]
    """
    def __init__(self):
        self.trace: List[Dict] = []

    def reset(self):
        self.trace = []

    def save_trace(self, filename="execution_trace.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.trace, f, indent=4, ensure_ascii=False)
        print(f"\nTrace successfully saved to {filename}")

    def load_trace(self, filename="execution_trace.json"):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)