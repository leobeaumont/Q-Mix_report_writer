from typing import List, Tuple
from .prompt_set import PromptSet
from .prompt_set_registry import PromptSetRegistry

@PromptSetRegistry.register("collector")
class CollectorPromptSet(PromptSet):
    @staticmethod
    def get_role():
        role = """
        # ROLE
        You are a Text Integration Engine. Your sole purpose is to clean and transition specific text blocks while removing all AI conversational filler.
        """
        return role
    
    @staticmethod
    def get_constraint(_role):
        constraint = """
        # STRICT RULES
        - Output **ONLY** the modified content from <current_text>.
        - Do NOT include the <previous_paragraph> in your final response.
        - Do NOT include any introductory or concluding remarks (zero AI poluting).
        - If the <current text> already flows well with the <previous_paragraph>, output the <current_text> exactly as provided (minus AI filler).
        """
        return constraint
    
    def get_description(self, _role):
        description = """
        # CONTEXT
        The text provided in the <current_text> tags has been meticulously reviewed and redacted by a full team of experts. Every word and redaction marker is intentional.

        # TASK INSTRUCTIONS
        1. **Remove Meta-Talk:** Delete all AI "discussion" sentences, introductory phrases (e.g., "Here is the redacted text:"), and closing questions.
        2. **Preserve Expert Content:** You are strictly forbidden from changing the meaning, tone, or specific wording of the expert-redacted content. Do not "fix" or alter any redaction markers.
        3. **Smooth Transition:** You may subtly adjust only the very beginning of the <current_text> to ensure a smooth logical flow from the <previous_paragraph>.
        4. **Output Constraint:** Your response must contain **ONLY** the modified <Current Text>.
        """
        return description

    def get_role_connection(self) -> List[Tuple[str, str]]:
        pass

    @staticmethod
    def get_format() -> str:
        pass

    @staticmethod
    def get_answer_prompt(question: str, role: str = "") -> str:
        pass

    @staticmethod
    def get_decision_constraint() -> str:
        pass

    @staticmethod
    def get_decision_role() -> str:
        pass