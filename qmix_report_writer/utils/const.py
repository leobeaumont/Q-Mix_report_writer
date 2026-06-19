import os
from pathlib import Path

# Directory of the installed/checked-out ``qmix_report_writer`` package. Bundled
# read-only resources (configs/, assets/) live here and travel with the package,
# so they resolve correctly whether run standalone or imported by a host project.
PACKAGE_ROOT = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

# Backwards-compatible alias. Historically PROJECT_ROOT pointed at the repo root;
# after the package move it points at the package dir, which is where the bundled
# configs/ and assets/ now live. Runtime data paths (chroma, output, traces) are
# resolved separately via the data root — see utils.config.get_data_root().
PROJECT_ROOT = PACKAGE_ROOT
