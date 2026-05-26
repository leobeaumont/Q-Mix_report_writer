"""
Base Node class for the multi-agent graph framework.

Each node represents a single agent in the Networked MMDP.
Nodes manage spatial (same-round) and temporal (cross-round) connections
and expose an execute() interface for LLM-based processing.
"""

import shortuuid
from typing import List, Any, Optional, Dict
from abc import ABC, abstractmethod
import asyncio

from utils.globals import PromptTokens, CompletionTokens, ReportState


class Node(ABC):
    """Base class for all agents in the QMIX multi-agent graph.

    Attributes:
        id: Unique identifier
        agent_name: Name of the agent type
        llm_name: LLM model name to use
        spatial_predecessors/successors: Same-round connections
        temporal_predecessors/successors: Cross-round connections
        outputs: Results from execution
        last_memory: Previous round's state for temporal processing
        token_usage: Tokens consumed by this node (used in QMIX observations)
    """

    def __init__(
        self,
        id: Optional[str] = None,
        agent_name: str = "",
        llm_name: str = "",
    ):
        self.id: str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name: str = agent_name
        self.llm_name: str = llm_name
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.role = ""
        self.last_memory: Dict[str, List[Any]] = {"inputs": [], "outputs": [], "raw_inputs": []}
        self.token_usage: int = 0

    @property
    def node_name(self):
        return self.__class__.__name__

    def add_predecessor(self, operation: "Node", st="spatial"):
        if st == "spatial" and operation not in self.spatial_predecessors:
            self.spatial_predecessors.append(operation)
            operation.spatial_successors.append(self)
        elif st == "temporal" and operation not in self.temporal_predecessors:
            self.temporal_predecessors.append(operation)
            operation.temporal_successors.append(self)

    def add_successor(self, operation: "Node", st="spatial"):
        if st == "spatial" and operation not in self.spatial_successors:
            self.spatial_successors.append(operation)
            operation.spatial_predecessors.append(self)
        elif st == "temporal" and operation not in self.temporal_successors:
            self.temporal_successors.append(operation)
            operation.temporal_predecessors.append(self)

    def remove_predecessor(self, operation: "Node", st="spatial"):
        if st == "spatial" and operation in self.spatial_predecessors:
            self.spatial_predecessors.remove(operation)
            operation.spatial_successors.remove(self)
        elif st == "temporal" and operation in self.temporal_predecessors:
            self.temporal_predecessors.remove(operation)
            operation.temporal_successors.remove(self)

    def remove_successor(self, operation: "Node", st="spatial"):
        if st == "spatial" and operation in self.spatial_successors:
            self.spatial_successors.remove(operation)
            operation.spatial_predecessors.remove(self)
        elif st == "temporal" and operation in self.temporal_successors:
            self.temporal_successors.remove(operation)
            operation.temporal_predecessors.remove(self)

    def clear_connections(self):
        self.spatial_predecessors = []
        self.spatial_successors = []
        self.temporal_predecessors = []
        self.temporal_successors = []

    def update_memory(self):
        self.last_memory["inputs"] = self.inputs
        self.last_memory["outputs"] = self.outputs
        self.last_memory["raw_inputs"] = self.raw_inputs

    def get_spatial_info(self) -> Dict[str, Dict]:
        spatial_info = {}
        if self.spatial_predecessors is not None:
            for predecessor in self.spatial_predecessors:
                pred_outputs = predecessor.outputs
                if isinstance(pred_outputs, list) and len(pred_outputs):
                    pred_output = pred_outputs[-1]
                elif isinstance(pred_outputs, list) and len(pred_outputs) == 0:
                    continue
                else:
                    pred_output = pred_outputs
                spatial_info[predecessor.id] = {"role": predecessor.role, "output": pred_output}
        return spatial_info

    def get_temporal_info(self) -> Dict[str, Any]:
        temporal_info = {}
        if self.temporal_predecessors is not None:
            for predecessor in self.temporal_predecessors:
                pred_outputs = predecessor.last_memory["outputs"]
                if isinstance(pred_outputs, list) and len(pred_outputs):
                    pred_output = pred_outputs[-1]
                elif isinstance(pred_outputs, list) and len(pred_outputs) == 0:
                    continue
                else:
                    pred_output = pred_outputs
                temporal_info[predecessor.id] = {"role": predecessor.role, "output": pred_output}
        return temporal_info

    def execute(self, input: Any, **kwargs):
        self.outputs = []
        self.token_usage = 0
        spatial_info = self.get_spatial_info()
        temporal_info = self.get_temporal_info()

        # Snapshot global token count before this node's LLM call
        tokens_before = PromptTokens.instance().value + CompletionTokens.instance().value

        results = [self._execute(input, spatial_info, temporal_info, **kwargs)]
        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)

        # token_usage = tokens consumed by this specific node's call
        self.token_usage = int(
            PromptTokens.instance().value + CompletionTokens.instance().value - tokens_before
        )
        return self.outputs

    async def async_execute(self, input: Any, **kwargs):
        self.outputs = []
        self.token_usage = 0
        spatial_info = self.get_spatial_info()
        temporal_info = self.get_temporal_info()

        # Snapshot global token count before this node's LLM call
        tokens_before = PromptTokens.instance().value + CompletionTokens.instance().value

        tasks = [asyncio.create_task(self._async_execute(input, spatial_info, temporal_info, **kwargs))]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)

        # token_usage = tokens consumed by this specific node's call
        self.token_usage = int(
            PromptTokens.instance().value + CompletionTokens.instance().value - tokens_before
        )
        return self.outputs

    def _build_user_prompt(
        self,
        raw_inputs: Dict,
        spatial_info: Dict,
        temporal_info: Dict,
        report_label: str,
        report_content: str,
        **kwargs,
    ) -> str:
        """Build the user-facing portion of the prompt from structured inputs.

        Shared by all agents. Each agent supplies its own report_label/content;
        the context block is delegated to the active prompt set so that the
        handcrafted graph and the QMIX graph can inject different metadata
        (phase info vs. selected action) without touching this method.
        """
        _TEMPORAL_MAX = 2000  # chars per prior-output entry to prevent context explosion
        spatial_str = "".join(
            f"#### Message from {info['role']}:\n{info['output']}\n\n"
            for info in spatial_info.values()
        )
        temporal_parts = []
        for info in temporal_info.values():
            out = str(info['output'])
            if len(out) > _TEMPORAL_MAX:
                out = out[:_TEMPORAL_MAX] + f"\n[...truncated — {len(info['output']) - _TEMPORAL_MAX} chars omitted]"
            temporal_parts.append(f"#### Previous output from {info['role']}:\n{out}\n\n")
        temporal_str = "".join(temporal_parts)

        user_prompt = f"\n\n### Report Subject:\n{raw_inputs['task']}\n"

        # Inject system-level context (phase for handcrafted; action for QMIX).
        prompt_set = getattr(self, "prompt_set", None)
        if prompt_set is not None:
            context_block = prompt_set.get_context_block(self.role, **kwargs)
            if context_block:
                user_prompt += f"\n{context_block}\n"

        user_prompt += f"\n### {report_label}:\n{report_content}\n"
        user_prompt += f"\n### Current Team Objective:\n{ReportState.instance().task}\n"

        if spatial_str:
            user_prompt += f"\n### Received messages:\n\n{spatial_str}"
        if temporal_str:
            user_prompt += f"### Your previous output:\n\n{temporal_str}"
        user_prompt += "### [WRITE OUTPUT HERE]"

        return user_prompt

    @abstractmethod
    def _execute(self, input: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        pass

    @abstractmethod
    async def _async_execute(self, input: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        pass

    @abstractmethod
    def _process_inputs(self, raw_inputs: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs) -> List[Any]:
        pass
