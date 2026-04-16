from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default='.', type=str)
    p.add_argument('--python', default=sys.executable, type=str)
    p.add_argument('--start', default='2000-01-31', type=str)
    p.add_argument('--end', default='2022-07-31', type=str)
    p.add_argument('--fallback-raw-dirs', default='', type=str)
    return p.parse_args()


def run() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    py = args.python
    steps = [
        ['scripts/01_prepare_macro_131.py', '--project-root', str(root), '--start', args.start, '--end', args.end, '--fallback-raw-dirs', args.fallback_raw_dirs],
        ['scripts/02_prepare_bond_data.py', '--project-root', str(root), '--start', args.start, '--end', args.end, '--fallback-raw-dirs', args.fallback_raw_dirs],
        ['scripts/03_run_replication.py', '--project-root', str(root)],
    ]
    for step in steps:
        subprocess.run([py] + step, cwd=root, check=True)


if __name__ == '__main__':
    run()
