"""Report-writing task list.

The benchmark dataset loaders (HumanEval / MMLU / GAIA / math / HLE / ...) were
inherited from the upstream Agent-Q-Mix problem-solving project and consumed
only by the deleted legacy QMIX eval. They were removed in the QMIX upgrade
(plan Stage 6.4, decision OD-3). Only the report-writing task list remains.
"""

from .tasks import tasks

__all__ = ["tasks"]
