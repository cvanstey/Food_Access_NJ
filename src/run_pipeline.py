"""
run_pipeline.py
================
Runs the full Food_Access_NJ pipeline in order, stopping immediately if
any stage fails. Each stage is run as its own subprocess (not imported),
so no changes to the individual stage scripts are required.

Usage:
    python run_pipeline.py              # run all stages
    python run_pipeline.py --from 03    # resume starting at 03_features.py
    python run_pipeline.py --only 04    # run a single stage
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


STAGES = [
    "01_load_data.py",
    "02a_nearest.py",
    "02b_merge_sources.py",
    "02c_clean_NJ_features_zip2.py",  # produces nj_zip_features_v2_clean.csv, required by 03_features.py
    "03_features.py",
    "04_model.py",
    "05_reports.py",
    "06_analytics.py",
    "../testing/compare_density.py",
    "07_targeted_analysis.py",
    #"08_zip_lookup.py", this is part of an interactive module. if you fork the repo, it should run as a script.
]


def stage_id(script_name: str) -> str:
    """First token before the underscore, e.g. '02a' from '02a_nearest.py'."""
    return script_name.split("_", 1)[0]


def run_stage(script_name: str, log_path: Path) -> tuple[bool, float]:
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        print(f"  [SKIP] {script_name} not found at {script_path}")
        return False, 0.0

    # Force UTF-8 everywhere: the log file itself, and the child process's
    # stdout encoding (via PYTHONIOENCODING). Without this, Windows falls
    # back to the console's active codepage (often cp1252), which doesn't
    # match the UTF-8 bytes emitted by print() statements containing
    # characters like — or ✔, producing mojibake (â€”, âœ”) in the log.
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"

    start = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 70}\n{script_name} — started {datetime.now().isoformat()}\n{'=' * 70}\n")
        log.flush()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=BASE_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env,
        )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  [FAIL] {script_name} exited with code {result.returncode} ({elapsed:.1f}s)")
        print(f"         See {log_path} for full output.")
        return False, elapsed

    print(f"  [OK]   {script_name} ({elapsed:.1f}s)")
    return True, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the Food_Access_NJ pipeline.")
    parser.add_argument("--from", dest="from_stage", default=None,
                         help="Resume starting at this stage id, e.g. 03")
    parser.add_argument("--only", dest="only_stage", default=None,
                         help="Run only this stage id, e.g. 04")
    args = parser.parse_args()

    stages_to_run = STAGES
    if args.only_stage:
        stages_to_run = [s for s in STAGES if stage_id(s) == args.only_stage]
        if not stages_to_run:
            print(f"No stage matches id '{args.only_stage}'")
            sys.exit(1)
    elif args.from_stage:
        ids = [stage_id(s) for s in STAGES]
        if args.from_stage not in ids:
            print(f"No stage matches id '{args.from_stage}'")
            sys.exit(1)
        start_idx = ids.index(args.from_stage)
        stages_to_run = STAGES[start_idx:]

    run_dir = BASE_DIR / "pipeline_logs"
    run_dir.mkdir(exist_ok=True)
    log_path = run_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print(f"Running {len(stages_to_run)} stage(s). Full output → {log_path}\n")

    total_start = time.time()
    for script in stages_to_run:
        print(f"→ {script}")
        ok, _ = run_stage(script, log_path)
        if not ok:
            print(f"\nPipeline stopped at {script}. Fix the error above and re-run with:")
            print(f"    python run_pipeline.py --from {stage_id(script)}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\nPipeline complete — {len(stages_to_run)} stage(s) in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()