"""
Compile a LaTeX report to PDF with zero manual setup.

There is no pure-Python TeX engine, so a real LaTeX binary is required.
**Tectonic** is the only supported engine. To keep the user from having to
install a multi-gigabyte TeX distribution, this module:

  1. Uses a ``tectonic`` already on PATH or in the git-ignored ``.tools/`` cache.
  2. Otherwise downloads the Tectonic single-file binary for the current
     OS / architecture into ``.tools/`` and uses that.

Tectonic is ideal here: it is one self-contained executable that fetches only
the LaTeX packages a document needs, on demand, and runs the multiple passes
required to resolve the table of contents and the ``thebibliography`` block by
itself. The first compilation needs internet access (to fetch the binary and
packages); afterwards everything is cached locally.

Entry point:
    from qmix_report_writer.utils.compile_pdf import compile_pdf
    pdf_path = compile_pdf("output/<run-folder>/report.tex")
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from .config import get_tools_dir
from .report_export import LATEX_NAME, PDF_NAME

# Local cache for the auto-downloaded Tectonic binary (git-ignored). Resolved
# lazily via config so a host can place it in a writable location rather than
# inside the (possibly read-only) installed package. See utils.config.get_tools_dir.

_GITHUB_LATEST = "https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest"

# Map (system, machine) -> the Rust target triple used in Tectonic asset names.
# Several Linux/macOS aliases map to the same triple.
_TARGET_TRIPLES = {
    ("windows", "amd64"): "x86_64-pc-windows-msvc",
    ("windows", "x86_64"): "x86_64-pc-windows-msvc",
    ("darwin", "x86_64"): "x86_64-apple-darwin",
    ("darwin", "arm64"): "aarch64-apple-darwin",
    ("darwin", "aarch64"): "aarch64-apple-darwin",
    ("linux", "x86_64"): "x86_64-unknown-linux-musl",
    ("linux", "amd64"): "x86_64-unknown-linux-musl",
    ("linux", "aarch64"): "aarch64-unknown-linux-musl",
    ("linux", "arm64"): "aarch64-unknown-linux-musl",
}

_INSTALL_HINT = (
    "Could not obtain a LaTeX engine. Install one of:\n"
    "  - Tectonic (recommended, single binary):\n"
    "      Windows: winget install TectonicProject.Tectonic\n"
    "      macOS:   brew install tectonic\n"
    "      Linux:   cargo install tectonic  (or your distro package)\n"
    "  - or a full TeX distribution (MiKTeX / TeX Live) providing pdflatex.\n"
    "Alternatively, ensure internet access so the Tectonic binary can be "
    "downloaded automatically into .tools/."
)


# ---------------------------------------------------------------------------
# Engine discovery
# ---------------------------------------------------------------------------

def _cached_tectonic() -> Optional[Path]:
    """Return the path to a previously downloaded Tectonic binary, if present."""
    exe = "tectonic.exe" if platform.system().lower() == "windows" else "tectonic"
    candidate = get_tools_dir() / exe
    return candidate if candidate.is_file() else None


def find_tectonic() -> Optional[str]:
    """Return a path to an existing Tectonic binary, or None.

    Checks the system ``PATH`` first, then the local ``.tools/`` download cache.
    Tectonic is the only supported engine; if none is found the caller downloads
    it via :func:`ensure_tectonic`.
    """
    on_path = shutil.which("tectonic")
    if on_path:
        return on_path
    cached = _cached_tectonic()
    if cached:
        return str(cached)
    return None


# ---------------------------------------------------------------------------
# Tectonic auto-download
# ---------------------------------------------------------------------------

def _target_triple() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    triple = _TARGET_TRIPLES.get((system, machine))
    if triple is None:
        raise RuntimeError(
            f"No Tectonic binary mapping for platform {system}/{machine}. "
            f"Please install Tectonic manually.\n\n{_INSTALL_HINT}"
        )
    return triple


def _select_asset(assets: List[dict], triple: str) -> Tuple[str, str]:
    """Pick the release asset matching this platform. Returns (name, url)."""
    # Tectonic publishes both a CLI archive and a "tectonic@<ver>" thin variant;
    # prefer a plain archive whose name carries the target triple.
    for asset in assets:
        name = asset.get("name", "")
        if triple in name and name.endswith((".zip", ".tar.gz")):
            return name, asset["browser_download_url"]
    raise RuntimeError(
        f"No Tectonic release asset found for target '{triple}'.\n\n{_INSTALL_HINT}"
    )


def _extract_binary(archive: Path, dest_dir: Path) -> Path:
    """Extract the tectonic executable from a downloaded archive into dest_dir."""
    exe = "tectonic.exe" if platform.system().lower() == "windows" else "tectonic"
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if Path(m).name == exe]
            if not members:
                raise RuntimeError(f"'{exe}' not found inside {archive.name}.")
            with zf.open(members[0]) as src, open(dest_dir / exe, "wb") as out:
                shutil.copyfileobj(src, out)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            members = [m for m in tf.getmembers() if Path(m.name).name == exe]
            if not members:
                raise RuntimeError(f"'{exe}' not found inside {archive.name}.")
            extracted = tf.extractfile(members[0])
            with open(dest_dir / exe, "wb") as out:
                shutil.copyfileobj(extracted, out)

    binary = dest_dir / exe
    # Mark executable on POSIX.
    if platform.system().lower() != "windows":
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return binary


def ensure_tectonic() -> Path:
    """Return a usable Tectonic binary, downloading it on first use if needed."""
    cached = _cached_tectonic()
    if cached:
        return cached

    tools_dir = get_tools_dir()
    tools_dir.mkdir(parents=True, exist_ok=True)
    triple = _target_triple()

    req = urllib.request.Request(_GITHUB_LATEST, headers={"User-Agent": "qmix-report-writer"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        release = json.load(resp)
    name, url = _select_asset(release.get("assets", []), triple)

    archive_path = tools_dir / name
    print(f"Downloading Tectonic ({name})…")
    dl = urllib.request.Request(url, headers={"User-Agent": "qmix-report-writer"})
    with urllib.request.urlopen(dl, timeout=120) as resp, open(archive_path, "wb") as out:
        shutil.copyfileobj(resp, out)

    binary = _extract_binary(archive_path, tools_dir)
    archive_path.unlink(missing_ok=True)
    print(f"Tectonic ready at {binary}")
    return binary


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _run(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def _build_command(engine: str, tex: Path, out_dir: Path) -> List[str]:
    """Return the Tectonic command to compile ``tex`` into ``out_dir``.

    Tectonic resolves the table of contents and the ``thebibliography``
    cross-references in a single invocation (it runs the required passes itself).
    """
    return [engine, "--keep-logs", "--outdir", str(out_dir), tex.name]


def compile_pdf(tex_path, out_dir=None) -> Path:
    """Compile a ``.tex`` file to PDF, returning the path to the produced PDF.

    Args:
        tex_path: Path to the ``.tex`` file (typically ``<run>/report.tex``).
        out_dir: Output directory for the PDF and intermediates. Defaults to the
            directory containing ``tex_path``.

    Raises:
        FileNotFoundError: if ``tex_path`` does not exist.
        RuntimeError: if Tectonic is unavailable or compilation fails.
    """
    tex_path = Path(tex_path).resolve()
    if not tex_path.is_file():
        raise FileNotFoundError(f"LaTeX source not found: {tex_path}")
    # Absolute, so the engine's --outdir is unaffected by the working directory.
    out_dir = (Path(out_dir).resolve() if out_dir is not None else tex_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tectonic is the only supported engine: use one on PATH or in the local
    # cache, otherwise download it automatically.
    engine = find_tectonic() or str(ensure_tectonic())

    cmd = _build_command(engine, tex_path, out_dir)
    last = _run(cmd, cwd=tex_path.parent)
    if last.returncode != 0:
        tail = (last.stdout or "")[-2000:] + "\n" + (last.stderr or "")[-1000:]
        raise RuntimeError(
            f"LaTeX compilation failed (tectonic, exit {last.returncode}).\n"
            f"--- output tail ---\n{tail.strip()}"
        )

    pdf_path = out_dir / (tex_path.stem + ".pdf")
    if not pdf_path.is_file():
        raise RuntimeError(
            f"Compilation reported success but no PDF was produced at {pdf_path}."
        )

    # Normalise the name to report.pdf for consistency with the run folder layout.
    final = out_dir / PDF_NAME
    if pdf_path.resolve() != final.resolve():
        shutil.move(str(pdf_path), str(final))
    return final


def compile_run_dir(run_dir) -> Path:
    """Compile ``report.tex`` inside a run folder into ``report.pdf``."""
    run_dir = Path(run_dir)
    return compile_pdf(run_dir / LATEX_NAME, out_dir=run_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile a report .tex to PDF.")
    parser.add_argument("target", help="Path to a .tex file or a run folder.")
    args = parser.parse_args()

    target = Path(args.target)
    tex = target if target.suffix == ".tex" else target / LATEX_NAME
    out = compile_pdf(tex)
    print(f"Wrote {out}")
