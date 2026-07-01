"""
Triggering logic for the PBDS tool.

The tool needs an Excel parameter workbook to work. It activates only when a
readable workbook exists at the configured path; when the file is absent the
factory returns None so callers leave the pipeline unchanged (a silent skip, no
exception).

The workbook path is configurable in configs/default.yaml (pbds.workbook_path)
and overloadable by a host exactly like the other paths — see
qmix_report_writer.utils.config.get_pbds_workbook_path for the resolution order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from qmix_report_writer.utils.config import get_active_pbds_workbook
from .pbds_manager import PBDSManager


def pbds_available() -> bool:
    """True when the configured PBDS workbook exists (i.e. the tool would activate)."""
    return get_active_pbds_workbook() is not None


def load_pbds_manager(
    workbook_path: Optional[Union[str, Path]] = None,
    default_k: int = 1,
) -> Optional[PBDSManager]:
    """Activate the PBDS tool, or return None when its workbook is unavailable.

    Args:
        workbook_path: an explicit workbook to use, overriding the configured path.
                       When None, the path is resolved from config/env via
                       get_active_pbds_workbook().
        default_k:     default hop radius handed to the PBDSManager.

    Returns:
        A PBDSManager bound to the workbook when a readable file exists, otherwise
        None — so the caller can skip the tool and leave the pipeline unchanged.
        A missing file never raises; it is a silent skip.
    """
    if workbook_path is not None:
        path: Optional[Path] = Path(workbook_path).expanduser()
        if not path.is_file():
            return None
    else:
        path = get_active_pbds_workbook()
        if path is None:
            return None
    return PBDSManager(str(path), default_k=default_k)
