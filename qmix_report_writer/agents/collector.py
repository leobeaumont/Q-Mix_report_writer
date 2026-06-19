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

# Sentinel outputs that carry no prose. The progress-summary LLM call is skipped
# for these (summarizing "[REMOVE_SECTION]" wastes a call), and they are never
# appended to the report.
_SENTINEL_OUTPUTS = frozenset({
    "[REMOVE_SECTION]",
    "[NO_REVISION_NEEDED]",
    "[HOLD]",
    "[NO SUPPORTED CONTENT]",
    "[NO_SUPPORTED_CONTENT]",
    "[NO NEW EVIDENCE]",
    "[WAITING_FOR_DIRECTIVE]",
})

from qmix_report_writer.graph.node import Node
from qmix_report_writer.agents.agent_registry import AgentRegistry
from qmix_report_writer.utils.config import get_llm
from qmix_report_writer.utils.globals import ReportState, SourceBuffer, strip_citation_tags
from qmix_report_writer.prompt.prompt_set_registry import PromptSetRegistry


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

    def _get_section_content(self) -> tuple:
        """Return (label, content) for the report section to show in the prompt.

        SECTION_REVIEW: show the exact section being revised so the Collector
        knows what it is replacing (not necessarily the last written section).
        All other phases: show the last written section as a continuation cue.
        """
        if self._is_revision_phase():
            idx = self.report.review_section_idx
            sections = self.report.sections
            if 0 <= idx < len(sections):
                # Strip citation tags so the Collector rewrites clean prose;
                # _apply_citation_tags() will re-tag the revised content afterwards.
                return "Current Section", strip_citation_tags(sections[idx]["content"])
            return "Current Section", "[No section content available]"
        return "Previous Text Production", self.report.get_last()

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = self.prompt_set.get_description(self.role)
        system_prompt += self.prompt_set.get_constraint(self.role)
        label, content = self._get_section_content()
        user_prompt = self._build_user_prompt(
            raw_inputs, spatial_info, temporal_info,
            label, content,
            **kwargs,
        )
        return system_prompt, user_prompt

    def _is_revision_phase(self) -> bool:
        try:
            from qmix_report_writer.handcrafted_graph.state import PhaseState
            from qmix_report_writer.handcrafted_graph.phases import PhaseType
            return PhaseState.instance().current_phase == PhaseType.SECTION_REVIEW
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
                if not line:
                    continue
                if not _ABSENCE_RE.search(line):
                    return True   # at least one positive evidence line
            return False          # DataAnalyst present but every line is an absence marker
        return True               # DataAnalyst not in spatial_info — don't block

    def _extract_section_id_from_review_index(self) -> str:
        """Resolve the target section ID from ReportState.review_section_idx.

        The index is set deterministically by the graph loop before each revision
        round — section targeting never depends on LLM output.
        """
        try:
            idx = self.report.review_section_idx
            sections = self.report.sections
            if 0 <= idx < len(sections):
                return sections[idx]["id"]
        except Exception:
            pass
        return ""

    def _removal_is_authorized(self) -> bool:
        """Return True when [REMOVE_SECTION] is backed by an actual instruction.

        Removal is destructive, so it requires provenance: either a validation
        directive (directive-bypass mode) or a Reviewer critique that explicitly
        asked for removal (flag set by the graph before the revision round).
        Without this guard a DataAnalyst that received no critique at all can
        hallucinate the sentinel and silently delete report content.
        """
        return bool(self.report.validation_directive) or bool(
            getattr(self.report, "removal_authorized", False)
        )

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        if not spatial_info:
            # Allow execution in directive-driven SECTION_REVIEW — the REVISION DIRECTIVE
            # is already in the context block; no Reviewer/DataAnalyst input is needed.
            if not (self._is_revision_phase() and self.report.validation_directive):
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

        clean = response1.strip()
        # The progress summary is only needed when real prose was produced —
        # summarizing a sentinel like [REMOVE_SECTION] wastes an LLM call.
        response2 = self.report.progress
        if clean and clean not in _SENTINEL_OUTPUTS:
            system_prompt, user_prompt = self._progress_prompt(self.report.progress, response1)
            message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            response2 = self.llm.gen(message, calling_agent="Collector")

        new_sources = self.source_buffer.flush()
        if self._is_revision_phase():
            # Section targeting is always resolved by the pipeline (review_section_idx),
            # never by parsing LLM output — this eliminates wrong-section replacements.
            section_id = self._extract_section_id_from_review_index()
            if not section_id:
                logger.error(
                    "SECTION_REVIEW: could not resolve section ID from review index — correction skipped."
                )
            elif clean == "[REMOVE_SECTION]":
                if not self._removal_is_authorized():
                    logger.error(
                        f"SECTION_REVIEW: [REMOVE_SECTION] for '{section_id}' has no backing "
                        f"directive or Reviewer removal instruction — section left unchanged."
                    )
                elif self.report.remove_section(section_id):
                    logger.info(f"SECTION_REVIEW: section '{section_id}' removed per directive.")
                else:
                    logger.error(f"SECTION_REVIEW: section '{section_id}' not found — removal skipped.")
            elif clean in _SENTINEL_OUTPUTS or not clean:
                # Collector echoed a sentinel instead of writing prose — leave section unchanged.
                logger.warning(
                    f"SECTION_REVIEW: Collector output sentinel '{clean}' instead of prose — section unchanged."
                )
            elif self.report.replace_section(section_id, response1, new_sources):
                self.report.progress = response2
            else:
                logger.error(
                    f"SECTION_REVIEW: section '{section_id}' not found in report — correction skipped."
                )
        elif not clean or clean in _SENTINEL_OUTPUTS:
            # No prose produced — never append an empty or sentinel-only section.
            logger.warning(
                f"Collector produced no section prose ({clean!r}) — nothing appended."
            )
            self.report.task = "[SECTION_SKIPPED — ASSIGN NEXT SECTION]"
        else:
            self.report.append(response1, response2, new_sources)
            self.report.task = "[SECTION_COMPLETE — ASSIGN NEXT SECTION]"
        if execution_trace:
            execution_trace.trace[-1]["Collector"]["report_state"] = self.report.content
        return response1

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        if not spatial_info:
            # Allow execution in directive-driven SECTION_REVIEW — the REVISION DIRECTIVE
            # is already in the context block; no Reviewer/DataAnalyst input is needed.
            if not (self._is_revision_phase() and self.report.validation_directive):
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

        clean = response1.strip()
        # The progress summary is only needed when real prose was produced —
        # summarizing a sentinel like [REMOVE_SECTION] wastes an LLM call.
        response2 = self.report.progress
        if clean and clean not in _SENTINEL_OUTPUTS:
            system_prompt, user_prompt = self._progress_prompt(self.report.progress, response1)
            message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            response2 = await self.llm.agen(message, calling_agent="Collector")

        new_sources = self.source_buffer.flush()
        if self._is_revision_phase():
            section_id = self._extract_section_id_from_review_index()
            if not section_id:
                logger.error(
                    "SECTION_REVIEW: could not resolve section ID from review index — correction skipped."
                )
            elif clean == "[REMOVE_SECTION]":
                if not self._removal_is_authorized():
                    logger.error(
                        f"SECTION_REVIEW: [REMOVE_SECTION] for '{section_id}' has no backing "
                        f"directive or Reviewer removal instruction — section left unchanged."
                    )
                elif self.report.remove_section(section_id):
                    logger.info(f"SECTION_REVIEW: section '{section_id}' removed per directive.")
                else:
                    logger.error(f"SECTION_REVIEW: section '{section_id}' not found — removal skipped.")
            elif clean in _SENTINEL_OUTPUTS or not clean:
                logger.warning(
                    f"SECTION_REVIEW: Collector output sentinel '{clean}' instead of prose — section unchanged."
                )
            elif self.report.replace_section(section_id, response1, new_sources):
                self.report.progress = response2
            else:
                logger.error(
                    f"SECTION_REVIEW: section '{section_id}' not found in report — correction skipped."
                )
        elif not clean or clean in _SENTINEL_OUTPUTS:
            # No prose produced — never append an empty or sentinel-only section.
            logger.warning(
                f"Collector produced no section prose ({clean!r}) — nothing appended."
            )
            self.report.task = "[SECTION_SKIPPED — ASSIGN NEXT SECTION]"
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
