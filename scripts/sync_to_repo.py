"""One-shot sync of source tree to the GitHub working copy.

Copies src/, docs/, results/, scripts/, README.md, requirements.txt from
this project to C:\\Users\\adm\\Downloads\\Term_paper_3rd_year.

Skips:
    - __pycache__/, .git/, .ipynb_checkpoints/
    - desktop.ini, .Rhistory, .DS_Store
    - .pyc / .pyo
    - data/  (too large for git)

Always overwrites at destination so this acts as a one-way sync.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

SRC = Path(r"C:\Users\adm\Desktop\Term Paper")
DST = Path(r"C:\Users\adm\Downloads\Term_paper_3rd_year")

EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
EXCLUDE_FILES = {"desktop.ini", ".Rhistory", ".DS_Store"}
EXCLUDE_EXTS = {".pyc", ".pyo"}


def should_skip(name: str) -> bool:
    if name in EXCLUDE_DIRS or name in EXCLUDE_FILES:
        return True
    if any(name.endswith(ext) for ext in EXCLUDE_EXTS):
        return True
    return False


def sync(src: Path, dst: Path) -> tuple[list[str], int]:
    copied: list[str] = []
    skipped_pyc = 0
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        dirs[:] = [d for d in dirs if not should_skip(d)]
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            if should_skip(f):
                if f.endswith(".pyc"):
                    skipped_pyc += 1
                continue
            sp = Path(root) / f
            tp = target_dir / f
            shutil.copy2(sp, tp)
            copied.append((rel / f).as_posix())
    return copied, skipped_pyc


def main() -> None:
    total_copied: list[str] = []
    total_skipped = 0
    for top in ["src", "docs", "results", "scripts", "tests", "config"]:
        s = SRC / top
        d = DST / top
        if not s.exists():
            continue
        c, sk = sync(s, d)
        total_copied.extend([f"{top}/{p}" for p in c])
        total_skipped += sk
        print(f"  {top}/  +{len(c)} files (skipped {sk} .pyc)")

    for f in ["README.md", "requirements.txt", "run_all.py"]:
        s = SRC / f
        if s.exists():
            shutil.copy2(s, DST / f)
            total_copied.append(f)
            print(f"  {f}  copied")

    print(f"\nTotal: {len(total_copied)} files synced; {total_skipped} .pyc skipped")


if __name__ == "__main__":
    main()
