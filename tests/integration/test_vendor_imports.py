"""tests/integration/test_vendor_imports.py — vendor smoke-import tests.

Verifies:
- vendor/node and vendor/shared are importable with mock stubs for heavy deps
- swarm-pin.txt exists, is a valid hex sha
- vendor/NOTICE.md has required fields
- forbidden dirs (tracker, client, scripts) are absent from vendor/
- peer_id invariant: agent.identity.compute_peer_id == shared.crypto.peer_id_from_public_key
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR_DIR = REPO_ROOT / "vendor"
SWARM_PIN_FILE = REPO_ROOT / "swarm-pin.txt"


# ---------------------------------------------------------------------------
# test_vendor_node_imports
# ---------------------------------------------------------------------------


class TestVendorNodeImports:
    """Smoke-import tests for vendor/node packages."""

    def test_import_node_package(self) -> None:
        """vendor/node/__init__.py is importable as `node`."""
        # Force reimport in case already cached without stubs
        sys.modules.pop("node", None)
        import node  # noqa: F401

        assert node is not None

    def test_import_node_weights(self) -> None:
        """node.weights is importable."""
        sys.modules.pop("node.weights", None)
        import node.weights  # noqa: F401

        assert node.weights is not None

    def test_import_node_weights_chunkstore(self) -> None:
        """node.weights.ChunkStore is importable and is a class."""
        sys.modules.pop("node.weights", None)
        from node.weights import ChunkStore

        assert isinstance(ChunkStore, type), f"Expected class, got {type(ChunkStore)}"

    def test_import_node_weights_weightmanager(self) -> None:
        """node.weights.WeightManager is importable and is a class."""
        sys.modules.pop("node.weights", None)
        from node.weights import WeightManager

        assert isinstance(WeightManager, type), f"Expected class, got {type(WeightManager)}"

    def test_import_node_inference(self) -> None:
        """node.inference is importable (with torch/transformers stubs)."""
        sys.modules.pop("node.inference", None)
        import node.inference  # noqa: F401

        assert node.inference is not None

    def test_import_node_inference_modelshard(self) -> None:
        """node.inference.ModelShard is importable and is a class."""
        sys.modules.pop("node.inference", None)
        from node.inference import ModelShard

        assert isinstance(ModelShard, type), f"Expected class, got {type(ModelShard)}"


# ---------------------------------------------------------------------------
# test_vendor_shared_imports
# ---------------------------------------------------------------------------


class TestVendorSharedImports:
    """Smoke-import tests for vendor/shared packages — no heavy dep mocks needed."""

    def test_import_shared_package(self) -> None:
        """vendor/shared/__init__.py is importable as `shared`."""
        import shared  # noqa: F401

        assert shared is not None

    def test_import_shared_crypto(self) -> None:
        """shared.crypto is importable."""
        import shared.crypto  # noqa: F401

        assert shared.crypto is not None

    def test_import_shared_crypto_sign_chunk_receipt(self) -> None:
        """shared.crypto.sign_chunk_receipt is a callable."""
        from shared.crypto import sign_chunk_receipt

        assert callable(sign_chunk_receipt)

    def test_import_shared_crypto_peer_id_from_public_key(self) -> None:
        """shared.crypto.peer_id_from_public_key is a callable."""
        from shared.crypto import peer_id_from_public_key

        assert callable(peer_id_from_public_key)

    def test_import_shared_protocol(self) -> None:
        """shared.protocol is importable."""
        import shared.protocol  # noqa: F401

        assert shared.protocol is not None

    def test_import_shared_protocol_forwardenvelope(self) -> None:
        """shared.protocol.ForwardEnvelope is importable."""
        from shared.protocol import ForwardEnvelope

        assert ForwardEnvelope is not None

    def test_import_shared_tls(self) -> None:
        """shared.tls is importable."""
        import shared.tls  # noqa: F401

        assert shared.tls is not None

    def test_import_shared_tls_error_class(self) -> None:
        """shared.tls.TLSPeerIdError is importable and is an Exception subclass."""
        from shared.tls import TLSPeerIdError

        assert issubclass(TLSPeerIdError, Exception)

    def test_import_shared_manifest(self) -> None:
        """shared.manifest is importable."""
        import shared.manifest  # noqa: F401

        assert shared.manifest is not None

    def test_import_shared_manifest_compute(self) -> None:
        """shared.manifest.compute_manifest_sha256 is a callable."""
        from shared.manifest import compute_manifest_sha256

        assert callable(compute_manifest_sha256)


# ---------------------------------------------------------------------------
# test_swarm_pin_present
# ---------------------------------------------------------------------------


class TestSwarmPin:
    """Tests for swarm-pin.txt."""

    def test_swarm_pin_exists(self) -> None:
        """swarm-pin.txt exists in repo root."""
        assert SWARM_PIN_FILE.exists(), f"swarm-pin.txt not found at {SWARM_PIN_FILE}"

    def test_swarm_pin_single_line(self) -> None:
        """swarm-pin.txt has exactly one non-empty line."""
        content = SWARM_PIN_FILE.read_text().strip()
        lines = [ln for ln in content.splitlines() if ln.strip()]
        assert len(lines) == 1, f"Expected 1 line in swarm-pin.txt, got {len(lines)}: {lines!r}"

    def test_swarm_pin_valid_sha(self) -> None:
        """swarm-pin.txt contains a valid hex SHA (at least 7 chars)."""
        sha = SWARM_PIN_FILE.read_text().strip()
        assert len(sha) >= 7, f"SHA too short: {sha!r}"
        assert re.fullmatch(r"[0-9a-f]+", sha), f"SHA is not valid lowercase hex: {sha!r}"

    def test_swarm_pin_is_40_chars(self) -> None:
        """swarm-pin.txt SHA is exactly 40 hex chars (full git SHA)."""
        sha = SWARM_PIN_FILE.read_text().strip()
        assert len(sha) == 40, f"Expected 40-char SHA, got {len(sha)}: {sha!r}"


# ---------------------------------------------------------------------------
# test_notice_present
# ---------------------------------------------------------------------------


class TestNoticeFile:
    """Tests for vendor/NOTICE.md."""

    def test_notice_exists(self) -> None:
        """vendor/NOTICE.md exists."""
        notice = VENDOR_DIR / "NOTICE.md"
        assert notice.exists(), f"vendor/NOTICE.md not found at {notice}"

    def test_notice_has_source_line(self) -> None:
        """vendor/NOTICE.md contains 'Source:' line."""
        content = (VENDOR_DIR / "NOTICE.md").read_text()
        assert "Source:" in content, "NOTICE.md missing 'Source:' line"

    def test_notice_has_commit_matching_pin(self) -> None:
        """vendor/NOTICE.md Commit: matches swarm-pin.txt."""
        sha = SWARM_PIN_FILE.read_text().strip()
        content = (VENDOR_DIR / "NOTICE.md").read_text()
        assert sha in content, (
            f"NOTICE.md does not contain the current pin SHA {sha!r}. "
            "Run `make vendor-sync` to regenerate."
        )

    def test_notice_has_license_mit(self) -> None:
        """vendor/NOTICE.md contains 'License: MIT'."""
        content = (VENDOR_DIR / "NOTICE.md").read_text()
        assert "License: MIT" in content, "NOTICE.md missing 'License: MIT' line"

    def test_notice_has_vendored_permission(self) -> None:
        """vendor/NOTICE.md contains vendored-with-permission statement."""
        content = (VENDOR_DIR / "NOTICE.md").read_text()
        assert "Vendored with permission" in content, (
            "NOTICE.md missing 'Vendored with permission' statement"
        )

    def test_notice_has_directories_line(self) -> None:
        """vendor/NOTICE.md lists node/ and shared/ directories."""
        content = (VENDOR_DIR / "NOTICE.md").read_text()
        assert "node/" in content and "shared/" in content, (
            "NOTICE.md missing directory listing (node/, shared/)"
        )


# ---------------------------------------------------------------------------
# test_no_forbidden_dirs
# ---------------------------------------------------------------------------


class TestNoForbiddenDirs:
    """Verify that vendor/ contains only whitelisted directories."""

    @pytest.mark.parametrize("forbidden", ["tracker", "client", "scripts"])
    def test_forbidden_dir_absent(self, forbidden: str) -> None:
        """vendor/{forbidden} must not exist."""
        path = VENDOR_DIR / forbidden
        assert not path.exists(), (
            f"vendor/{forbidden} must not exist — only node/ and shared/ are whitelisted. "
            "Check sync_vendor.py whitelist enforcement."
        )

    def test_vendor_contains_node_and_shared(self) -> None:
        """vendor/ contains node/ and shared/ directories."""
        assert (VENDOR_DIR / "node").is_dir(), "vendor/node/ missing"
        assert (VENDOR_DIR / "shared").is_dir(), "vendor/shared/ missing"


# ---------------------------------------------------------------------------
# test_peer_id_invariant
# ---------------------------------------------------------------------------


class TestPeerIdInvariant:
    """Verify peer_id derivation is consistent between agent and vendor shared."""

    def test_peer_id_invariant_matches_vendor(self) -> None:
        """peer_id from agent.identity.compute_peer_id == shared.crypto.peer_id_from_public_key.

        Both must implement the same invariant:
            peer_id = base58(sha256(public_key_bytes))
        """
        # Check vendor shared has peer_id_from_public_key
        try:
            from shared.crypto import peer_id_from_public_key as vendor_peer_id
        except (ImportError, AttributeError) as e:
            pytest.fail(
                f"vendor must export peer_id_from_public_key — "
                f"base58(sha256(pubkey)) invariant broken: {e}"
            )

        # Check agent identity
        try:
            from agent.identity import compute_peer_id as agent_peer_id
        except ImportError:
            pytest.skip(reason="agent.identity not available")

        # Generate a deterministic test key (32 bytes)
        import hashlib

        # Use a fixed seed to produce a deterministic public key for the test
        test_pubkey = hashlib.sha256(b"test-peer-id-invariant-fixture-seed").digest()
        assert len(test_pubkey) == 32, "test public key must be 32 bytes"

        agent_result = agent_peer_id(test_pubkey)
        vendor_result = vendor_peer_id(test_pubkey)

        assert agent_result == vendor_result, (
            f"peer_id invariant mismatch:\n"
            f"  agent.identity.compute_peer_id  = {agent_result!r}\n"
            f"  shared.crypto.peer_id_from_public_key = {vendor_result!r}\n"
            "Both must implement: peer_id = base58(sha256(pubkey))"
        )

    def test_peer_id_formula_is_base58_sha256(self) -> None:
        """Verify peer_id formula: base58(sha256(pubkey)) against known vector."""
        import hashlib

        import base58

        pubkey = hashlib.sha256(b"test-peer-id-invariant-fixture-seed").digest()
        expected = base58.b58encode(hashlib.sha256(pubkey).digest()).decode()

        try:
            from shared.crypto import peer_id_from_public_key as vendor_peer_id
        except (ImportError, AttributeError) as e:
            pytest.fail(
                f"vendor must export peer_id_from_public_key — "
                f"base58(sha256(pubkey)) invariant broken: {e}"
            )

        vendor_result = vendor_peer_id(pubkey)
        assert vendor_result == expected, (
            f"vendor peer_id_from_public_key does not implement base58(sha256(pubkey)):\n"
            f"  expected = {expected!r}\n"
            f"  got      = {vendor_result!r}"
        )


# ---------------------------------------------------------------------------
# Whitelist and audit unit tests (inline, no sync side effects)
# ---------------------------------------------------------------------------


class TestSyncVendorLogic:
    """Unit tests for sync_vendor.py whitelist and audit logic."""

    def test_whitelist_passes_for_node_paths(self) -> None:
        """_check_whitelist passes for node/ paths."""
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from sync_vendor import _check_whitelist  # type: ignore[import]
        finally:
            if str(REPO_ROOT / "tools") in sys.path:
                sys.path.remove(str(REPO_ROOT / "tools"))

        # Should not raise
        _check_whitelist([Path("node/__init__.py"), Path("node/weights.py")])
        _check_whitelist([Path("shared/crypto.py"), Path("shared/protocol.py")])

    def test_whitelist_rejects_tracker_paths(self) -> None:
        """_check_whitelist raises ValueError for tracker/ paths."""
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from sync_vendor import _check_whitelist  # type: ignore[import]
        finally:
            if str(REPO_ROOT / "tools") in sys.path:
                sys.path.remove(str(REPO_ROOT / "tools"))

        with pytest.raises(ValueError, match="outside whitelist"):
            _check_whitelist([Path("tracker/main.py")])

    def test_whitelist_rejects_client_paths(self) -> None:
        """_check_whitelist raises ValueError for client/ paths."""
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from sync_vendor import _check_whitelist  # type: ignore[import]
        finally:
            if str(REPO_ROOT / "tools") in sys.path:
                sys.path.remove(str(REPO_ROOT / "tools"))

        with pytest.raises(ValueError, match="outside whitelist"):
            _check_whitelist([Path("client/sdk.py")])

    def test_whitelist_rejects_scripts_paths(self) -> None:
        """_check_whitelist raises ValueError for scripts/ paths."""
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from sync_vendor import _check_whitelist  # type: ignore[import]
        finally:
            if str(REPO_ROOT / "tools") in sys.path:
                sys.path.remove(str(REPO_ROOT / "tools"))

        with pytest.raises(ValueError, match="outside whitelist"):
            _check_whitelist([Path("scripts/deploy.sh")])

    def test_audit_clean_on_normal_code(self, tmp_path: Path) -> None:
        """_audit_files returns empty list for clean code."""
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from sync_vendor import _audit_files  # type: ignore[import]
        finally:
            if str(REPO_ROOT / "tools") in sys.path:
                sys.path.remove(str(REPO_ROOT / "tools"))

        (tmp_path / "node").mkdir()
        (tmp_path / "node" / "clean.py").write_text(
            "# Normal Python code\ndef foo(): pass\n"
        )
        violations = _audit_files(tmp_path)
        assert violations == [], f"Expected no violations, got: {violations}"

    def test_audit_detects_private_key_header(self, tmp_path: Path) -> None:
        """_audit_files detects '-----BEGIN RSA PRIVATE KEY' pattern."""
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from sync_vendor import _audit_files  # type: ignore[import]
        finally:
            if str(REPO_ROOT / "tools") in sys.path:
                sys.path.remove(str(REPO_ROOT / "tools"))

        (tmp_path / "node").mkdir()
        (tmp_path / "node" / "bad.py").write_text(
            '# leaked key\nKEY = "-----BEGIN RSA PRIVATE KEY-----"\n'
        )
        violations = _audit_files(tmp_path)
        assert len(violations) >= 1, "Expected at least one violation for PRIVATE KEY"

    def test_audit_detects_aws_key(self, tmp_path: Path) -> None:
        """_audit_files detects AWS access key pattern AKIA..."""
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from sync_vendor import _audit_files  # type: ignore[import]
        finally:
            if str(REPO_ROOT / "tools") in sys.path:
                sys.path.remove(str(REPO_ROOT / "tools"))

        (tmp_path / "shared").mkdir()
        (tmp_path / "shared" / "bad.py").write_text(
            'AWS_KEY = "AKIAIOSFODNN7EXAMPLE"\n'
        )
        violations = _audit_files(tmp_path)
        assert len(violations) >= 1, "Expected at least one violation for AWS key"

    def test_audit_detects_sudri_internal(self, tmp_path: Path) -> None:
        """_audit_files detects 'sudri-internal' pattern."""
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from sync_vendor import _audit_files  # type: ignore[import]
        finally:
            if str(REPO_ROOT / "tools") in sys.path:
                sys.path.remove(str(REPO_ROOT / "tools"))

        (tmp_path / "node").mkdir()
        (tmp_path / "node" / "bad.py").write_text(
            'URL = "https://sudri-internal.corp/api"\n'
        )
        violations = _audit_files(tmp_path)
        assert len(violations) >= 1, "Expected violation for sudri-internal"

    def test_audit_does_not_flag_stdlib_secrets_ref(self, tmp_path: Path) -> None:
        """_audit_files does NOT flag 'secrets.token_bytes' (Python stdlib reference)."""
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            from sync_vendor import _audit_files  # type: ignore[import]
        finally:
            if str(REPO_ROOT / "tools") in sys.path:
                sys.path.remove(str(REPO_ROOT / "tools"))

        (tmp_path / "shared").mkdir()
        (tmp_path / "shared" / "ok.py").write_text(
            "import secrets\ntoken = secrets.token_bytes(32)\n"
        )
        violations = _audit_files(tmp_path)
        assert violations == [], (
            f"stdlib secrets.token_bytes should NOT trigger audit, got: {violations}"
        )


# ---------------------------------------------------------------------------
# Webclient tokens.css vendor
# ---------------------------------------------------------------------------


class TestWebclientVendor:
    """Tests for vendor/tokens.css (webclient sync --target webclient)."""

    def test_vendor_tokens_css_exists(self) -> None:
        """vendor/tokens.css must exist after sync."""
        tokens_css = VENDOR_DIR / "tokens.css"
        assert tokens_css.exists(), (
            "vendor/tokens.css not found. Run `make sync-tokens` to populate it."
        )

    def test_vendor_tokens_css_nonempty(self) -> None:
        """vendor/tokens.css must be non-empty."""
        tokens_css = VENDOR_DIR / "tokens.css"
        if not tokens_css.exists():
            pytest.skip("vendor/tokens.css not present")
        assert tokens_css.stat().st_size > 0, "vendor/tokens.css is empty"

    def test_webclient_pin_exists(self) -> None:
        """webclient-pin.txt must exist after sync."""
        webclient_pin = REPO_ROOT / "webclient-pin.txt"
        assert webclient_pin.exists(), (
            "webclient-pin.txt not found. Run `make sync-tokens` to populate it."
        )

    def test_webclient_pin_valid_sha(self) -> None:
        """webclient-pin.txt must be a valid 40-char hex SHA."""
        webclient_pin = REPO_ROOT / "webclient-pin.txt"
        if not webclient_pin.exists():
            pytest.skip("webclient-pin.txt not present")
        sha = webclient_pin.read_text().strip()
        assert len(sha) == 40, f"Expected 40-char SHA in webclient-pin.txt, got {len(sha)}: {sha!r}"
        assert re.fullmatch(r"[0-9a-f]+", sha), (
            f"webclient-pin.txt SHA is not valid lowercase hex: {sha!r}"
        )

    def test_vendor_tokens_css_parseable(self) -> None:
        """vendor/tokens.css must be parseable by parse_tokens without error."""
        tokens_css = VENDOR_DIR / "tokens.css"
        if not tokens_css.exists():
            pytest.skip("vendor/tokens.css not present")

        import importlib.util as _ilu

        spec_path = REPO_ROOT / "tools" / "build_qss.py"
        spec = _ilu.spec_from_file_location("build_qss_vendor_smoke", spec_path)
        assert spec is not None and spec.loader is not None
        module = _ilu.module_from_spec(spec)
        sys.modules["build_qss_vendor_smoke"] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            css = tokens_css.read_text(encoding="utf-8")
            tokens = module.parse_tokens(css)
            module.resolve_all_colors(tokens)
            module.validate_required_tokens(tokens)
        finally:
            sys.modules.pop("build_qss_vendor_smoke", None)

    def test_sync_vendor_webclient_idempotent(self, tmp_path: Path) -> None:
        """sync_webclient() with same SHA is a no-op (idempotency check via mock git archive).

        Uses monkeypatching of module-level constants rather than subprocess.
        """
        import importlib.util as _ilu

        spec_path = REPO_ROOT / "tools" / "sync_vendor.py"
        spec = _ilu.spec_from_file_location("sync_vendor_test_idem", spec_path)
        assert spec is not None and spec.loader is not None
        mod = _ilu.module_from_spec(spec)
        sys.modules["sync_vendor_test_idem"] = mod
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]

            # Point WEBCLIENT_PIN_FILE at a tmp file pre-filled with the "current" sha
            test_sha = "a" * 40
            tmp_pin = tmp_path / "webclient-pin.txt"
            tmp_pin.write_text(test_sha + "\n")

            # Patch constants on the freshly loaded module
            mod.WEBCLIENT_PIN_FILE = tmp_pin  # type: ignore[attr-defined]
            mod.WEBCLIENT_REPO = REPO_ROOT  # point at existing dir to pass existence check

            # Capture stdout to confirm no-op message
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                mod.sync_webclient(commit=test_sha)

            output = buf.getvalue()
            assert "No-op" in output, (
                f"Expected 'No-op' in output for idempotent sync, got: {output!r}"
            )
        finally:
            sys.modules.pop("sync_vendor_test_idem", None)
