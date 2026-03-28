"""HLE (Humanity's Last Exam) prompt set for agent_q_mix.

Inherits AgenticPromptSet but widens the answer range beyond A-J
since HLE MCQ can have options up to V.
"""

from .prompt_set_registry import PromptSetRegistry
from .agentic_prompt_set import AgenticPromptSet


@PromptSetRegistry.register("hle")
class HLEPromptSet(AgenticPromptSet):

    @staticmethod
    def get_constraint(role=None):
        return """
            I will ask you a very challenging question.
            I will give you multiple answers enumerated as A, B, C, D, E, and possibly more.
            Only one answer out of the offered options is correct.
            You must choose the correct answer to the question.
            Your response must be the letter corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you. Please think critically and not just follow the answers of the majority.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter corresponding to the correct option.
        """

    @staticmethod
    def get_analyze_constraint(role=None):
        return """
I will ask you a very challenging question with multiple answer options.
Only one answer is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents.
Your reply must be less than 100 words but include your answer and a brief step by step analysis.
The first line of your reply must contain only one letter corresponding to the correct option.
"""

    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a very challenging question with multiple answer options.
        Only one answer is correct.
        You must choose the correct answer to the question.
        Your response must be the letter corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        """
