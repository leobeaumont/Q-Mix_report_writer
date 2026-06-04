from graph.node import Node
from agents.agent_registry import AgentRegistry
from utils.config import get_llm
from utils.globals import ReportState
from prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("Reviewer")
class Reviewer(Node):
    """Text reviewer of the team, here to ensure the quality of the production."""

    def __init__(self, id=None, role=None, llm_name=""):
        super().__init__(id, "Reviewer", llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get("redacting")
        self.role = role or "Reviewer"
        self.report = ReportState.instance()

    def _is_section_review_phase(self) -> bool:
        try:
            from handcrafted_graph.state import PhaseState
            from handcrafted_graph.phases import PhaseType
            return PhaseState.instance().current_phase == PhaseType.SECTION_REVIEW
        except Exception:
            return False

    def _is_validation_phase(self) -> bool:
        try:
            from handcrafted_graph.state import PhaseState
            from handcrafted_graph.phases import PhaseType
            return PhaseState.instance().current_phase == PhaseType.VALIDATION
        except Exception:
            return False

    def _get_review_content(self) -> tuple:
        """Return (report_label, report_content) appropriate for the current phase.

        SECTION_REVIEW: single section text — no truncation needed.
        VALIDATION window review: the sections in the current window (complete text).
        VALIDATION synthesis: a compact section-list overview — full content not needed
            since findings are passed through get_context_block validation_notes.
        All other phases: full report with head+tail cap for context safety.
        """
        if self._is_section_review_phase():
            idx = self.report.review_section_idx
            sections = self.report.sections
            if 0 <= idx < len(sections):
                section = sections[idx]
                header = f"Section ID: {section['id']} | Title: {section['title'] or '(untitled)'}\n\n"
                return "Section Content", header + section["content"]
            return "Section Content", "[No section content available]"

        if self._is_validation_phase():
            window_info = self.report.validation_window
            if window_info is not None:
                # Window review — show only the sections in this window
                i, n_windows, window_sections = window_info
                content = "\n\n".join(
                    f"### {s['title'] or s['id']}\n{s['content']}"
                    for s in window_sections
                )
                return f"Report Window {i + 1} of {n_windows}", content
            else:
                # Synthesis round — full content already reviewed; show structure only
                return "Report Structure", self.report.list_sections(verbose=True)

        # Fallback: head + tail truncation for any legacy path
        _MAX_CHARS = 6000
        _HALF = _MAX_CHARS // 2
        full = self.report.content or self.report.progress
        if len(full) <= _MAX_CHARS:
            return "Full Report Content", full
        head = full[:_HALF]
        tail = full[-_HALF:]
        omitted = len(full) - _MAX_CHARS
        return (
            "Full Report Content",
            head + f"\n\n[...{omitted} chars omitted — showing head and tail...]\n\n" + tail,
        )

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = self.prompt_set.get_description(self.role)
        system_prompt += self.prompt_set.get_constraint(self.role)
        report_label, report_content = self._get_review_content()
        user_prompt = self._build_user_prompt(
            raw_inputs, spatial_info, temporal_info,
            report_label, report_content,
            **kwargs,
        )
        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        if execution_trace:
            execution_trace.trace[-1]["Reviewer"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm.gen(message, calling_agent="Reviewer")
        if execution_trace:
            execution_trace.trace[-1]["Reviewer"]["response"] = response
        return response

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        execution_trace = kwargs.get("execution_trace", None)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        if execution_trace:
            execution_trace.trace[-1]["Reviewer"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = await self.llm.agen(message, calling_agent="Reviewer")
        if execution_trace:
            execution_trace.trace[-1]["Reviewer"]["response"] = response
        return response

if __name__ == "__main__":
    input_arg = {"task": "write a report"}
    spatial = {"1": {"role": "Other Agent", "output": "Message from other agent"}}
    temporal = {"0": {"role": "This agent", "output": "Message from last round"}}
    col = Reviewer(llm_name="tinyllama")

    system, user = col._process_inputs(input_arg, spatial, temporal)

    print(system)
    print("=" * 60)
    print(user)
