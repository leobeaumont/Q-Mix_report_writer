from typing import List, Optional, Dict

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
        self.additions = []
        self.sources = []
        self.progress = "[NOTHING WRITTEN SO FAR]"
    
    def reset(self):
        self.content = ""
        self.additions = []
        self.sources = []
        self.progress = "[NOTHING WRITTEN SO FAR]"

    def append(self, text: str, progress: str, new_sources: Optional[List] = None):
        self.content += text
        self.additions.append(text)
        self.progress = progress

        if new_sources is not None:
            self.sources += new_sources
    
    def get_last(self) -> str:
        if len(self.additions) > 0:
            return self.additions[-1]
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
