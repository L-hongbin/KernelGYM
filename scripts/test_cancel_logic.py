"""Run cancellation/lifecycle regressions without connecting to a deployed service.

The lifecycle tests own a disposable Unix-socket Redis server (no TCP listener,
no persistence). This replaces the obsolete marker-only in-memory test harness.
"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/server/test_task_manager_resource_queues.py",
            "tests/server/test_workflow_lifecycle.py",
            *sys.argv[1:],
        ],
        cwd=root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
