"""tools/sync_vendor.py — vendor sync for llm-swarm node/ and shared/.

CLI:
    uv run python tools/sync_vendor.py [--commit <sha>]

If --commit is omitted, takes HEAD from ../llm-swarm/.
Uses `git archive` — never touches the ../llm-swarm working tree.

Whitelist: node/** and shared/** only.
Audit: sensitive pattern grep before writing to vendor/.
Smoke: import test with stub stubs for heavy deps.
Atomic: vendor/.tmp/ staging, then os.replace into vendor/.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SWARM_REPO = REPO_ROOT.parent / "llm-swarm"
VENDOR_DIR = REPO_ROOT / "vendor"
SWARM_PIN_FILE = REPO_ROOT / "swarm-pin.txt"

WHITELIST_DIRS: list[str] = ["node", "shared"]

# Audit patterns use word-boundaries / anchors to avoid false positives on
# stdlib references like "secrets.token_bytes" (Python stdlib) or "password_hash".
# Each pattern targets actual credential literals, not API names.
AUDIT_PATTERNS: list[str] = [
    r"(?i)\b(secret_key|api_secret|client_secret|shared_secret)\b",
    r"(?i)\bpassword\s*=\s*['\"][^'\"]{4,}",
    r"-----BEGIN (RSA |EC |OPENSSH |)PRIVATE KEY",
    r"AKIA[0-9A-Z]{16}",
    r"AIza[0-9A-Za-z_\-]{35}",
    r"(?<!\w)10\.\d{1,3}\.\d{1,3}\.\d{1,3}(?!\w)",
    r"192\.168\.\d{1,3}\.\d{1,3}",
    r"sudri-internal",
    r"\.sudri\.ru.*\b(internal|staging|dev)\b",
]

# Heavy deps that need stubs during smoke import
HEAVY_DEPS: list[str] = [
    "torch",
    "bitsandbytes",
    "aioquic",
    "pynvml",
    "transformers",
    "nacl",
]

NOTICE_TEMPLATE = """\
# Vendor NOTICE
Source: sudri-code/llm-swarm (private)
Commit: {commit}
Date: {date}
Directories: node/, shared/
License: MIT (inherited from llm-swarm-desktop LICENSE)
Vendored with permission of the upstream repository owner (sudri-code).
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
    """Run a subprocess, quoting args defensively."""
    return subprocess.run(args, check=True, capture_output=True, **kwargs)  # type: ignore[call-overload]


def _get_head_sha() -> str:
    result = _run(["git", "-C", str(SWARM_REPO), "rev-parse", "HEAD"])
    return result.stdout.decode().strip()


def _read_current_pin() -> str | None:
    if SWARM_PIN_FILE.exists():
        content = SWARM_PIN_FILE.read_text().strip()
        return content if content else None
    return None


def _check_whitelist(paths: list[Path]) -> None:
    """Raise ValueError if any path is outside the whitelist dirs."""
    for p in paths:
        parts = p.parts
        if not parts:
            continue
        # parts[0] is the top-level directory in the archive (e.g. "node", "shared")
        top = parts[0]
        if top not in WHITELIST_DIRS:
            raise ValueError(
                f"Path {p} is outside whitelist {WHITELIST_DIRS!r}. "
                "Only node/ and shared/ are allowed. Sync aborted."
            )


def _audit_files(staging_dir: Path) -> list[tuple[Path, int, str]]:
    """Return list of (file, lineno, pattern) for any sensitive matches."""
    compiled = [(re.compile(pat, re.IGNORECASE), pat) for pat in AUDIT_PATTERNS]
    violations: list[tuple[Path, int, str]] = []
    for py_file in sorted(staging_dir.rglob("*")):
        if not py_file.is_file():
            continue
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for compiled_re, pat in compiled:
                if compiled_re.search(line):
                    violations.append((py_file.relative_to(staging_dir), lineno, pat))
    return violations


def _generate_stub_dir(stub_dir: Path) -> None:
    """Generate minimal stub modules for heavy deps in stub_dir."""
    stubs: dict[str, str] = {
        "torch/__init__.py": """\
# stub
import types

class Tensor: pass
class nn:
    class Module: pass
    class Linear: pass
    class Embedding: pass
    class LayerNorm: pass
    class Parameter: pass
    class ModuleList: pass
    class Sequential: pass

def zeros(*args, **kwargs): return Tensor()
def ones(*args, **kwargs): return Tensor()
def tensor(*args, **kwargs): return Tensor()
def load(*args, **kwargs): return {}
def save(*args, **kwargs): pass
def no_grad(): return _NullCtx()
def inference_mode(): return _NullCtx()
def use_deterministic_algorithms(*args, **kwargs): pass
def amp(*args, **kwargs): pass
def device(*args): return object()
def cuda(*args, **kwargs):
    return types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        get_device_properties=lambda d: types.SimpleNamespace(total_memory=0),
        mem_get_info=lambda d=None: (0, 0),
    )
def backends(*args, **kwargs): return object()

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __call__(self, fn): return fn

float16 = "float16"
float32 = "float32"
bfloat16 = "bfloat16"
int8 = "int8"
long = "long"
bool = "bool"

cuda = types.SimpleNamespace(
    is_available=lambda: False,
    device_count=lambda: 0,
    get_device_properties=lambda d: types.SimpleNamespace(total_memory=0, major=0, minor=0),
    mem_get_info=lambda d=None: (0, 0),
    empty_cache=lambda: None,
    synchronize=lambda: None,
)
backends = types.SimpleNamespace(
    cuda=types.SimpleNamespace(
        matmul=types.SimpleNamespace(allow_tf32=False),
        allow_tf32=False,
        deterministic=False,
        benchmark=False,
    ),
    cudnn=types.SimpleNamespace(deterministic=False, benchmark=False),
)
mps = types.SimpleNamespace(is_available=lambda: False)
""",
        "torch/nn/__init__.py": """\
from torch import nn as _nn
Module = _nn.Module
Linear = _nn.Linear
Embedding = _nn.Embedding
LayerNorm = _nn.LayerNorm
Parameter = _nn.Parameter
ModuleList = _nn.ModuleList
Sequential = _nn.Sequential
""",
        "transformers/__init__.py": """\
# stub
class AutoConfig:
    @classmethod
    def from_pretrained(cls, *args, **kwargs): return object()

class LlamaForCausalLM:
    pass

class PretrainedConfig:
    pass
""",
        "transformers/cache_utils.py": """\
class DynamicCache:
    pass
""",
        "transformers/models/__init__.py": "",
        "transformers/models/llama/__init__.py": "",
        "transformers/models/llama/modeling_llama.py": """\
class LlamaDecoderLayer:
    pass
class LlamaRMSNorm:
    pass
class LlamaRotaryEmbedding:
    pass
""",
        "bitsandbytes/__init__.py": """\
class nn:
    class Linear8bitLt:
        pass
""",
        "bitsandbytes/nn/__init__.py": """\
class Linear8bitLt:
    pass
""",
        "aioquic/__init__.py": "",
        "aioquic/asyncio/__init__.py": "",
        "aioquic/asyncio/client.py": "",
        "aioquic/asyncio/server.py": "",
        "aioquic/quic/__init__.py": "",
        "aioquic/quic/configuration.py": """\
class QuicConfiguration:
    def __init__(self, *args, **kwargs): pass
""",
        "aioquic/quic/connection.py": "",
        "aioquic/quic/events.py": """\
class StreamDataReceived: pass
class StreamReset: pass
class ConnectionTerminated: pass
""",
        "pynvml/__init__.py": """\
def nvmlInit(): pass
def nvmlShutdown(): pass
def nvmlDeviceGetCount(): return 0
def nvmlDeviceGetHandleByIndex(i): return object()
def nvmlDeviceGetMemoryInfo(handle): return type('M', (), {'total': 0, 'free': 0, 'used': 0})()
def nvmlDeviceGetTemperature(handle, sensor): return 0
NVMLError = Exception
NVML_TEMPERATURE_GPU = 0
""",
        "nacl/__init__.py": "",
        "nacl/exceptions.py": """\
class BadSignatureError(Exception): pass
""",
        "nacl/signing.py": """\
class SigningKey:
    def __init__(self, seed=None): pass
    @property
    def verify_key(self): return VerifyKey()
    def sign(self, message): return b''
    @classmethod
    def generate(cls): return cls()

class VerifyKey:
    def __init__(self, key=None): pass
    def verify(self, smessage, signature=None): pass
""",
    }

    for rel_path, content in stubs.items():
        dest = stub_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)


def _smoke_import(staging_dir: Path) -> tuple[bool, str]:
    """Run smoke import in subprocess with stub stubs.

    Returns (success, error_message).
    """
    with tempfile.TemporaryDirectory(prefix="swarm_stubs_") as stub_dir_str:
        stub_dir = Path(stub_dir_str)
        _generate_stub_dir(stub_dir)

        # Build PYTHONPATH: stubs first, then staging_dir (node/ and shared/ live here)
        python_path = os.pathsep.join([str(stub_dir), str(staging_dir)])
        env = {**os.environ, "PYTHONPATH": python_path}

        smoke_code = (
            "from node.weights import ChunkStore; "
            "from node.inference import ModelShard; "
            "from shared.crypto import sign_chunk_receipt; "
            "from shared.protocol import ForwardEnvelope; "
            "from shared.tls import TLSPeerIdError; "
            "from shared.manifest import compute_manifest_sha256; "
            "print('smoke ok')"
        )

        result = subprocess.run(
            [sys.executable, "-c", smoke_code],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr.strip()


# ---------------------------------------------------------------------------
# Main sync logic
# ---------------------------------------------------------------------------


def sync(commit: str | None = None) -> None:
    """Perform vendor sync."""
    if not SWARM_REPO.is_dir():
        print(f"ERROR: ../llm-swarm not found at {SWARM_REPO}", file=sys.stderr)
        sys.exit(1)

    # 1. Resolve commit sha
    if commit is None:
        sha = _get_head_sha()
        print(f"No --commit specified; using HEAD: {sha[:12]}...")
    else:
        sha = commit.strip()
        print(f"Using specified commit: {sha[:12]}...")

    # 2. Check idempotency
    current_pin = _read_current_pin()
    if current_pin == sha:
        print(f"Already at {sha[:12]}; vendor is up to date. No-op.")
        return

    # 3. Extract via git archive into temp dir
    print("Extracting via git archive (not touching ../llm-swarm working tree)...")
    with tempfile.TemporaryDirectory(prefix="swarm_archive_") as archive_dir_str:
        archive_dir = Path(archive_dir_str)
        archive_file = archive_dir / "archive.tar"

        # git -C <repo> archive <sha> node shared > archive.tar
        archive_args = [
            "git", "-C", str(SWARM_REPO),
            "archive", sha, "node", "shared",
        ]
        with archive_file.open("wb") as f:
            subprocess.run(archive_args, check=True, stdout=f, stderr=subprocess.PIPE)

        # Extract into staging dir
        staging_dir = archive_dir / "staging"
        staging_dir.mkdir()
        with tarfile.open(archive_file) as tf:
            members = tf.getmembers()
            # 4. Whitelist check
            paths_in_archive = [Path(m.name) for m in members if m.name]
            try:
                _check_whitelist(paths_in_archive)
            except ValueError as e:
                print(f"ERROR: Whitelist violation: {e}", file=sys.stderr)
                sys.exit(1)

            tf.extractall(staging_dir, filter="data")  # noqa: S202 — controlled source (own git repo)

        # 5. Audit
        print("Running sensitive-pattern audit...")
        violations = _audit_files(staging_dir)
        if violations:
            print("ERROR: Sensitive patterns found. Sync aborted.", file=sys.stderr)
            for file_path, lineno, pattern in violations:
                print(f"  {file_path}:{lineno}  pattern={pattern!r}", file=sys.stderr)
            sys.exit(1)
        print(f"Audit clean ({len(list(staging_dir.rglob('*.py')))} .py files checked).")

        # 6. Smoke import
        print("Running smoke import test...")
        ok, err = _smoke_import(staging_dir)
        if not ok:
            print("ERROR: Smoke import failed. swarm-pin.txt will NOT be updated.", file=sys.stderr)
            print(err, file=sys.stderr)
            sys.exit(1)
        print("Smoke import: OK.")

        # 7. Build new vendor content in a temp staging area alongside vendor/
        tmp_vendor = VENDOR_DIR.parent / "vendor.tmp_sync"
        if tmp_vendor.exists():
            shutil.rmtree(tmp_vendor)
        tmp_vendor.mkdir()

        # Copy whitelist dirs from staging into tmp_vendor
        for wl_dir in WHITELIST_DIRS:
            src = staging_dir / wl_dir
            if src.exists():
                shutil.copytree(src, tmp_vendor / wl_dir)
                print(f"  Copied {wl_dir}/ ({sum(1 for _ in src.rglob('*.py'))} .py files)")

        # Preserve existing vendor/README.md if present
        readme_src = VENDOR_DIR / "README.md"
        if readme_src.exists():
            shutil.copy2(readme_src, tmp_vendor / "README.md")

        # Write NOTICE.md
        now_utc = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        notice_content = NOTICE_TEMPLATE.format(commit=sha, date=now_utc)
        (tmp_vendor / "NOTICE.md").write_text(notice_content)

        # 8. Atomic replace: vendor/ ← vendor.tmp_sync/
        # Strategy: rename existing vendor/node and vendor/shared to
        # vendor/.old_node / vendor/.old_shared, move new ones in from
        # tmp_vendor, then remove .old_* backups.  If the rename into
        # vendor/ fails we attempt to restore from .old_* before raising.
        # vendor/README.md is left untouched throughout.
        old_dirs: list[tuple[Path, Path]] = []
        try:
            # Step 1 — back up existing whitelist dirs by rename (fast, same fs)
            for wl_dir in WHITELIST_DIRS:
                dest_wl = VENDOR_DIR / wl_dir
                old_wl = VENDOR_DIR / f".old_{wl_dir}"
                if old_wl.exists():
                    shutil.rmtree(old_wl)
                if dest_wl.exists():
                    dest_wl.rename(old_wl)
                    old_dirs.append((old_wl, dest_wl))

            # Step 2 — move new dirs into vendor/ by rename
            for wl_dir in WHITELIST_DIRS:
                src_wl = tmp_vendor / wl_dir
                dest_wl = VENDOR_DIR / wl_dir
                if src_wl.exists():
                    src_wl.rename(dest_wl)

            # Step 3 — write NOTICE.md and clean up .old_* backups
            (VENDOR_DIR / "NOTICE.md").write_text(notice_content)
            for old_wl, _ in old_dirs:
                if old_wl.exists():
                    shutil.rmtree(old_wl)

        except Exception:
            # Recovery: restore backups if step 2 partially failed
            for old_wl, dest_wl in old_dirs:
                if old_wl.exists() and not dest_wl.exists():
                    with contextlib.suppress(OSError):
                        old_wl.rename(dest_wl)  # best-effort; leave .old_* for manual recovery
            raise

        # Clean up tmp staging dir
        if tmp_vendor.exists():
            shutil.rmtree(tmp_vendor)

    # 9. Write swarm-pin.txt atomically
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=REPO_ROOT,
        prefix=".swarm-pin.",
        suffix=".tmp",
        delete=False,
    ) as tmp_pin:
        tmp_pin.write(sha + "\n")
        tmp_pin_path = tmp_pin.name
    os.replace(tmp_pin_path, SWARM_PIN_FILE)

    print(f"\nSync complete. Pin: {sha}")
    print("  vendor/NOTICE.md  written")
    print(f"  swarm-pin.txt     = {sha}")

    # Summary
    node_dir = VENDOR_DIR / "node"
    shared_dir = VENDOR_DIR / "shared"
    node_count = sum(1 for _ in node_dir.rglob("*.py")) if node_dir.exists() else 0
    shared_count = sum(1 for _ in shared_dir.rglob("*.py")) if shared_dir.exists() else 0
    print(f"  vendor/node/      {node_count} .py files")
    print(f"  vendor/shared/    {shared_count} .py files")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync vendor/ from ../llm-swarm")
    parser.add_argument(
        "--commit",
        metavar="SHA",
        default=None,
        help="Git commit SHA to archive (default: HEAD of ../llm-swarm)",
    )
    args = parser.parse_args()
    sync(commit=args.commit)


if __name__ == "__main__":
    main()
