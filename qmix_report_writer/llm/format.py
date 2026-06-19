import dataclasses
from typing import Literal

MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message:
    role: MessageRole
    content: str

    def to_dict(self):
        return dataclasses.asdict(self)
