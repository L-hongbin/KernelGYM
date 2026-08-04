"""Build and verify the CUPTI TSC timestamp shim (LD_PRELOAD).

The shim (kernelgym/native/cupti_tsc_shim.cpp) works around the CUDA
12.6u2-13.0 CUPTI bug where Kineto's TSC timestamp callback produces start=0
kernel records that Kineto then drops, leaving torch.profiler with zero CUDA
kernels. See docs/design-doc/PROFILER_EMPTY_CAPTURE.md.

Deployment contract:

- ``KERNELGYM_CUPTI_TSC_SHIM=true`` (profile env) asks the service CLI to build
  the shim and inject ``LD_PRELOAD`` + ``KINETO_TSC_FIXED=true`` +
  ``KERNELGYM_CUPTI_TSC_SHIM_EXPECTED=<path>`` into every service process. If
  the build fails, nothing is injected and the legacy multi-forward profiling
  workaround stays active (fail-open).
- At runtime, after a profiler context has run in a process, the shim's actual
  state is queryable; when a shim is expected but did not engage, callers must
  fall back to the legacy workaround instead of trusting single-forward
  profiling (``kineto_tsc_fix_verified``).
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SHIM_SOURCE = PROJECT_ROOT / "kernelgym" / "native" / "cupti_tsc_shim.cpp"
SHIM_BUILD_DIR = PROJECT_ROOT / ".native"

SHIM_FLAG_ENV = "KERNELGYM_CUPTI_TSC_SHIM"
SHIM_EXPECTED_ENV = "KERNELGYM_CUPTI_TSC_SHIM_EXPECTED"

# Mirror of ShimState in cupti_tsc_shim.cpp.
STATE_NOT_CALLED = 0
STATE_ENGAGED_NATIVE = 1
STATE_PASSTHROUGH_FIXED = 2
STATE_PASSTHROUGH_ERROR = 3
STATE_FAILED = 4

_HEALTHY_STATES = (STATE_ENGAGED_NATIVE, STATE_PASSTHROUGH_FIXED)


def _source_digest() -> str:
    return hashlib.sha256(SHIM_SOURCE.read_bytes()).hexdigest()[:12]


def shim_artifact_path() -> Path:
    return SHIM_BUILD_DIR / f"libkernelgym_cupti_tsc_shim-{_source_digest()}.so"


def ensure_shim_built() -> Optional[Path]:
    """Compile the shim if its artifact for the current source is missing.

    Returns the artifact path, or None when the build is impossible or fails
    (callers must fall back to the legacy profiling workaround).
    """
    try:
        artifact = shim_artifact_path()
    except OSError as exc:
        logger.error("CUPTI TSC shim source unreadable: %s", exc)
        return None
    if artifact.exists():
        return artifact
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        logger.error("CUPTI TSC shim build skipped: no C++ compiler on PATH")
        return None
    SHIM_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=SHIM_BUILD_DIR, suffix=".so.tmp", delete=False) as handle:
        temp_path = Path(handle.name)
    command = [
        compiler,
        "-O2",
        "-fPIC",
        "-shared",
        "-Wall",
        str(SHIM_SOURCE),
        "-o",
        str(temp_path),
        "-ldl",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if completed.returncode != 0:
            logger.error(
                "CUPTI TSC shim build failed (%s): %s",
                completed.returncode,
                (completed.stderr or "").strip()[-2000:],
            )
            return None
        temp_path.chmod(0o755)
        temp_path.replace(artifact)
    except Exception as exc:
        logger.error("CUPTI TSC shim build failed: %s", exc)
        return None
    finally:
        temp_path.unlink(missing_ok=True)
    logger.info("CUPTI TSC shim built: %s", artifact)
    return artifact


def expected_shim_path() -> Optional[str]:
    return os.environ.get(SHIM_EXPECTED_ENV) or None


def shim_state() -> Optional[int]:
    """Current shim state in this process, or None when no shim is loaded."""
    try:
        lib = ctypes.CDLL(None)
        return int(lib.kernelgym_cupti_tsc_shim_state())
    except (OSError, AttributeError):
        return None


def shim_cupti_version() -> Optional[int]:
    try:
        lib = ctypes.CDLL(None)
        return int(lib.kernelgym_cupti_tsc_shim_cupti_version())
    except (OSError, AttributeError):
        return None


def shim_state_healthy(state: Optional[int]) -> bool:
    return state in _HEALTHY_STATES


def kineto_tsc_fix_verified(kineto_tsc_fixed: bool) -> Optional[bool]:
    """Whether the declared Kineto TSC fix is actually active in this process.

    Only meaningful after a profiler context has run (Kineto registers the
    timestamp callback on first GPU tracing). Returns None when no fix is
    declared, True when the declaration is trusted (custom Kineto build, no
    shim expected) or the shim engaged, and False when a shim was expected but
    is missing or failed to engage — callers must then fall back to the legacy
    multi-forward workaround.
    """
    if not kineto_tsc_fixed:
        return None
    if expected_shim_path() is None:
        return True
    return shim_state_healthy(shim_state())
