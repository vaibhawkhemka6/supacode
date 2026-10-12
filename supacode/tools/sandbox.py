"""What is physically possible, underneath what permissions.py allows.

permissions.py is a cooperative convention: it decides what's worth
interrupting a human for, but it never touches the filesystem or a
subprocess itself, and nothing stops a bug (or a future call site) from
calling `TOOLS_BY_NAME[name].func(**args)` directly, skipping `call_tool`
and the permission gate with it entirely. This module is the layer
underneath that gap - it lives *inside* the tool functions themselves
(registry.py calls it directly), so it holds even when the gate above it
is bypassed, and even when a human said "y" to something a human
shouldn't have been able to approve in the first place.

Two concrete guarantees, matching the two risky primitives the tools
touch:
- `confine(path)`: a write can physically not land outside PROJECT. Not
  "ask first" - no yes makes this succeed. Read access is untouched;
  read_file was already unconditionally allowed at the policy layer
  (no side effects worth gating), and confining it too would make the
  agent unable to inspect anything outside its own project, which costs
  more than it protects.
- `run_sandboxed(command)`: a subprocess can physically not exhaust the
  host - capped CPU time, capped process count, and (Linux only - see
  CAP_ADDRESS_SPACE below) capped address space - on top of the
  wall-clock timeout bash() already had. A forkbomb dies on its own
  limits on every POSIX platform; a memory leak dies on its own limits
  on Linux, and on the wall-clock timeout everywhere else.
"""

import subprocess
import sys
from pathlib import Path

try:
    import resource  # POSIX only - no equivalent on Windows
except ImportError:
    resource = None

PROJECT = Path.cwd().resolve()

CPU_SECONDS = 30
MAX_BYTES = 1 << 30  # 1 GiB address space - Linux only, see _cap_resources
MAX_PROCS = 64

# RLIMIT_AS is unenforceable on macOS: dyld maps its shared cache as one
# sparse region that counts against "address space used" regardless of
# actual memory touched, so even a bare `echo` fails exec under any
# RLIMIT_AS low enough to matter (verified: 1 GiB through 64 GiB all
# fail identically). Capping it there would break every command, not
# just the ones worth capping - so it's Linux-only, not "best effort
# everywhere." CPU time and process count don't have this problem and
# are capped on every POSIX platform.
CAP_ADDRESS_SPACE = resource is not None and sys.platform != "darwin"


class SandboxError(Exception):
    """Raised for something the sandbox makes impossible, not just unapproved."""


def confine(path: str) -> Path:
    """Resolve `path` and guarantee it lands inside PROJECT, or raise.

    `.resolve()` both follows symlinks and collapses `..`, so neither a
    symlink planted inside the project nor a `../../etc/passwd`-style
    traversal can land outside PROJECT without tripping this - there's
    no unresolved-looking path that slips through.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved != PROJECT and PROJECT not in resolved.parents:
        raise SandboxError(
            f"{path} resolves to {resolved}, outside the sandboxed root {PROJECT}"
        )
    return resolved


def _cap_resources():
    """preexec_fn: runs in the child, after fork, before exec - these
    caps apply to the subprocess tree bash() starts, never to supacode
    itself (which stays in the parent process, uncapped).
    """
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCS, MAX_PROCS))
    if CAP_ADDRESS_SPACE:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_BYTES, MAX_BYTES))


def run_sandboxed(command: str, timeout: int) -> subprocess.CompletedProcess:
    """Same call bash() always made, with resource caps underneath the
    timeout it already had. `preexec_fn` is only passed where `resource`
    actually exists - subprocess.run refuses it outright on Windows.
    """
    kwargs = {"preexec_fn": _cap_resources} if resource else {}
    return subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=timeout, **kwargs
    )
