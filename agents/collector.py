import logging
import re

logger = logging.getLogger("handcrafted_graph.collector")

# Patterns that mark an evidence item as absent/unavailable from the knowledge base.
# Used by _data_analyst_has_content() to decide whether to silence the Collector.
_ABSENCE_RE = re.compile(
    r'\b(?:State Deficiency|absent|not found|no definition|no data|unavailable|Term absent)\b'
    r'|\[NO NEW EVIDENCE\]|\[WAITING_FOR_DIRECTIVE\]|\[NO SUPPORTED CONTENT\]',
    re.IGNORECASE,
)

from graph.node import Node
from agents.agent_registry import AgentRegistry
from utils.config import get_llm
from utils.globals import ReportState, SourceBuffer
from prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("Collector")
class Collector(Node):
    """Collector agent meant to collect report text as it is written."""

    def __init__(self, id=None, role=None, llm_name=""):
        super().__init__(id, "Collector", llm_name)
        self.llm = get_llm(llm_name)
        self.prompt_set = PromptSetRegistry.get("redacting")
        self.role = role or "Collector"
        self.report = ReportState.instance()
        self.source_buffer = SourceBuffer.instance()

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = self.prompt_set.get_description(self.role)
        system_prompt += self.prompt_set.get_constraint(self.role)
        user_prompt = self._build_user_prompt(
            raw_inputs, spatial_info, temporal_info,
            "Previous Text Production", self.report.get_last(),
            **kwargs,
        )
        return system_prompt, user_prompt

    def _is_revision_phase(self) -> bool:
        try:
            from handcrafted_graph.state import PhaseState
            from handcrafted_graph.phases import PhaseType
            return PhaseState.instance().current_phase == PhaseType.REVISION
        except Exception:
            return False

    def _data_analyst_has_content(self, spatial_info: dict) -> bool:
        """Return False when DataAnalyst's message consists entirely of absence markers.

        Triggers a Collector skip so that meta-commentary about missing evidence
        never reaches the report. When DataAnalyst is not present in spatial_info
        the method returns True (don't block other senders).
        """
        for info in spatial_info.values():
            if info.get("role") != "Data Analyst":
                continue
            output = str(info.get("output", ""))
            for line in output.splitlines():
                line = line.strip()
                if not line or line.startswith("[SECTION_ID:"):
                    continue
                if not _ABSENCE_RE.search(line):
                    return True   # at least one positive evidence line
            return False          # DataAnalyst present but every line is an absence marker
        return True               # DataAnalyst not in spatial_info — don't block

    def _extract_section_id(self, spatial_info: dict) -> str:
        """Parse [SECTION_ID: section_X] tag emitted by DataAnalyst during REVISION."""
        for info in spatial_info.values():
            match = re.search(
                r'\[SECTION_ID:\s*(section_\d+)\]',
                str(info.get("output", "")),
                re.IGNORECASE,
            )
            if match:
                return match.group(1)
        return ""

    def _extract_section_id_from_task(self) -> str:
        """Fallback: parse a section_X ID from the current LeadArchitect task directive.

        Used when DataAnalyst omits the mandatory [SECTION_ID] tag. The LeadArchitect
        sets ReportState.task to a directive such as 'correct section_2', so a
        section_N pattern is reliably present when the REVISION phase is on track.
        """
        match = re.search(r"section_\d+", self.report.task, re.IGNORECASE)
        return match.group(0).lower() if match else ""

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        if not spatial_info:  # If no agent appended, the collector stays idle
            return
        if not self._data_analyst_has_content(spatial_info):
            if not self._is_revision_phase():
                self.report.task = "[SECTION_SKIPPED — ASSIGN NEXT SECTION]"
            return
        execution_trace = kwargs.get("execution_trace", None)

        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response1 = self.llm.gen(message, calling_agent="Collector")
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["response"] = response1

        previous_progress = self.report.progress
        system_prompt, user_prompt = self._progress_prompt(previous_progress, response1)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response2 = self.llm.gen(message, calling_agent="Collector")

        new_sources = self.source_buffer.flush()
        if self._is_revision_phase():
            section_id = self._extract_section_id(spatial_info)
            if not section_id:
                section_id = self._extract_section_id_from_task()
                if section_id:
                    logger.warning(
                        "REVISION: DataAnalyst omitted [SECTION_ID] tag; "
                        f"falling back to '{section_id}' from task directive."
                    )
                else:
                    logger.error(
                        "REVISION: DataAnalyst omitted [SECTION_ID] tag and no section ID "
                        "could be inferred from the task directive — correction skipped."
                    )
            if section_id and self.report.replace_section(section_id, response1):
                self.report.progress = response2
                self.report.task = "[CORRECTION_APPLIED — ASSIGN NEXT CORRECTION]"
            elif section_id:
                logger.error(
                    f"REVISION: section '{section_id}' not found in report — correction skipped."
                )
        else:
            self.report.append(response1, response2, new_sources)
            self.report.task = "[SECTION_COMPLETE — ASSIGN NEXT SECTION]"
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["report_state"] = self.report.content
        return response1

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        if not spatial_info:  # If no agent appended, the collector stays idle
            return
        if not self._data_analyst_has_content(spatial_info):
            if not self._is_revision_phase():
                self.report.task = "[SECTION_SKIPPED — ASSIGN NEXT SECTION]"
            return
        execution_trace = kwargs.get("execution_trace", None)

        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["prompt"] = system_prompt + user_prompt
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response1 = await self.llm.agen(message, calling_agent="Collector")
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["response"] = response1

        previous_progress = self.report.progress
        system_prompt, user_prompt = self._progress_prompt(previous_progress, response1)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response2 = await self.llm.agen(message, calling_agent="Collector")

        new_sources = self.source_buffer.flush()
        if self._is_revision_phase():
            section_id = self._extract_section_id(spatial_info)
            if not section_id:
                section_id = self._extract_section_id_from_task()
                if section_id:
                    logger.warning(
                        "REVISION: DataAnalyst omitted [SECTION_ID] tag; "
                        f"falling back to '{section_id}' from task directive."
                    )
                else:
                    logger.error(
                        "REVISION: DataAnalyst omitted [SECTION_ID] tag and no section ID "
                        "could be inferred from the task directive — correction skipped."
                    )
            if section_id and self.report.replace_section(section_id, response1):
                self.report.progress = response2
                self.report.task = "[CORRECTION_APPLIED — ASSIGN NEXT CORRECTION]"
            elif section_id:
                logger.error(
                    f"REVISION: section '{section_id}' not found in report — correction skipped."
                )
        else:
            self.report.append(response1, response2, new_sources)
            self.report.task = "[SECTION_COMPLETE — ASSIGN NEXT SECTION]"
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["report_state"] = self.report.content
        return response1

    def _progress_prompt(self, previous_progress: str, addition: str) -> str:
        system_prompt = self.prompt_set.get_description("Summarizer")

        user_prompt = f"\n\n### Previous summary:\n{previous_progress}\n"
        user_prompt += f"\n### New addition:\n{addition}\n"
        user_prompt += "### [WRITE NEW SUMMARY HERE]"

        return system_prompt, user_prompt
    
if __name__ == "__main__":
    spatial = {"key": "This is the current text"}
    col = Collector(llm_name="tinyllama")

    system, user = col._process_inputs([], spatial, {})

    print(system)
    print("=" * 60)
    print(user)

    old = "Old report state."
    new = "New addition."

    system, user = col._progress_prompt(old, new)
    print("\n\n")
    print(system)
    print("=" * 60)
    print(user)
