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
        print(f"Trace successfully saved to {filename}")

    def load_trace(self, filename="execution_trace.json"):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)