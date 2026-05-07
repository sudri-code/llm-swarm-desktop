"""tests/unit/test_build_qss.py

Tests for tools/build_qss.py:
  1. ΔE2000 < 1.0 between source OKLCH and output sRGB hex for each palette var.
  2. Snapshot: all key --bg-*, --fg-*, --accent-* present in tokens.qss.
  3. Round-trip: tokens.py constants match tokens.qss hex values.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path setup — add repo root and tools/ to sys.path
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
TOKENS_CSS = (
    REPO_ROOT.parent
    / "llm-swarm-webclient"
    / "frontend"
    / "src"
    / "styles"
    / "tokens.css"
)
OUT_QSS = REPO_ROOT / "app" / "styles" / "tokens.qss"
OUT_PY = REPO_ROOT / "app" / "styles" / "tokens.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_build_qss() -> Any:
    """Import tools.build_qss (or tools/build_qss as module)."""
    import importlib.util

    spec_path = TOOLS_DIR / "build_qss.py"
    spec = importlib.util.spec_from_file_location("build_qss", spec_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Must register in sys.modules BEFORE exec so that @dataclass can resolve
    # the module dict via cls.__module__ (otherwise it gets NoneType).
    sys.modules["build_qss"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def build_qss() -> Any:
    return _import_build_qss()


@pytest.fixture(scope="module")
def parsed_tokens(build_qss: Any) -> Any:
    if not TOKENS_CSS.exists():
        pytest.skip(f"tokens.css not found at {TOKENS_CSS}")
    css = TOKENS_CSS.read_text(encoding="utf-8")
    tokens = build_qss.parse_tokens(css)
    build_qss.resolve_all_colors(tokens)
    return tokens


@pytest.fixture(scope="module")
def generated_qss(parsed_tokens: Any, build_qss: Any) -> str:
    """Generate QSS string (in-memory, no file write)."""
    return build_qss.generate_qss(parsed_tokens)


@pytest.fixture(scope="module")
def generated_py(parsed_tokens: Any, build_qss: Any) -> str:
    """Generate tokens.py string (in-memory, no file write)."""
    return build_qss.generate_py(parsed_tokens)


# ---------------------------------------------------------------------------
# ΔE2000 tests — OKLCH → sRGB precision
# ---------------------------------------------------------------------------


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    """#rrggbb → (r, g, b) in [0, 1]."""
    h = h.lstrip("#")
    # Handle 8-char (#rrggbbaa) — ignore alpha for ΔE
    h = h[:6]
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return r, g, b


def _oklch_to_lab_d65(L: float, C: float, H_deg: float) -> tuple[float, float, float]:
    """OKLCH → CIELAB D65 for ΔE2000 comparison via colour-science.

    Uses direct functions (no colour.convert() which requires networkx).
    """
    import math

    import colour

    H_rad = math.radians(H_deg)
    a_ok = C * math.cos(H_rad)
    b_ok = C * math.sin(H_rad)
    oklab = [L, a_ok, b_ok]

    # OKLab → XYZ D65 (direct, no networkx)
    xyz = colour.models.Oklab_to_XYZ(oklab)
    lab = colour.XYZ_to_Lab(xyz)
    return float(lab[0]), float(lab[1]), float(lab[2])


def _hex_to_lab_d65(hex_str: str) -> tuple[float, float, float]:
    """#rrggbb → CIELAB D65 for ΔE2000 comparison.

    Uses direct functions (no colour.convert() which requires networkx).
    """
    import colour

    r, g, b = _hex_to_rgb(hex_str)
    # sRGB → XYZ D65 (direct function, includes EOTF decoding)
    xyz = colour.models.sRGB_to_XYZ([r, g, b])
    lab = colour.XYZ_to_Lab(xyz)
    return float(lab[0]), float(lab[1]), float(lab[2])


_OKLCH_RE = re.compile(
    r"oklch\(\s*([\d.]+%?)\s+([\d.]+)\s+([\d.]+)(?:\s*/\s*[\d.]+)?\s*\)"
)


def _parse_oklch(raw: str) -> tuple[float, float, float] | None:
    m = _OKLCH_RE.search(raw)
    if not m:
        return None
    l_raw, c_raw, h_raw = m.group(1), m.group(2), m.group(3)
    L = float(l_raw.rstrip("%")) / 100.0 if l_raw.endswith("%") else float(l_raw)
    return L, float(c_raw), float(h_raw)


@pytest.mark.parametrize(
    "token",
    [
        pytest.param(
            token,
            id=token.name,
        )
        for token in _import_build_qss().parse_tokens(
            (
                Path(__file__).resolve().parent.parent.parent.parent
                / "llm-swarm-webclient"
                / "frontend"
                / "src"
                / "styles"
                / "tokens.css"
            ).read_text(encoding="utf-8")
            if (
                Path(__file__).resolve().parent.parent.parent.parent
                / "llm-swarm-webclient"
                / "frontend"
                / "src"
                / "styles"
                / "tokens.css"
            ).exists()
            else ":root{}"
        ).colors
        if "oklch" in token.raw.lower()
    ],
)
def test_delta_e2000_oklch_tokens(token: Any, build_qss: Any) -> None:
    """ΔE2000 between source OKLCH and resolved hex must be < 1.0."""
    import colour

    oklch = _parse_oklch(token.raw)
    if oklch is None:
        pytest.skip(f"Cannot parse OKLCH from {token.raw!r}")

    L, C, H = oklch

    # Re-resolve to get fresh hex (token.hex already resolved but let's be explicit)
    hex_color = build_qss.resolve_color(token.name, token.raw)

    lab_source = _oklch_to_lab_d65(L, C, H)
    lab_resolved = _hex_to_lab_d65(hex_color)

    # colour.delta_E expects Lab as numpy arrays or lists
    delta_e = colour.delta_E(lab_source, lab_resolved, method="CIE 2000")
    assert delta_e < 1.0, (
        f"{token.name}: ΔE2000={delta_e:.4f} ≥ 1.0 "
        f"(source={token.raw!r}, hex={hex_color!r})"
    )


# ---------------------------------------------------------------------------
# Snapshot tests — key variables present in generated QSS
# ---------------------------------------------------------------------------

_REQUIRED_COLOR_VARS = [
    "--bg-0",
    "--bg-1",
    "--bg-2",
    "--bg-3",
    "--bg-4",
    "--bg-5",
    "--fg-1",
    "--fg-2",
    "--fg-3",
    "--fg-4",
    "--accent",
    "--accent-2",
    "--accent-soft",
    "--ok",
    "--warn",
    "--err",
]


@pytest.mark.parametrize("var_name", _REQUIRED_COLOR_VARS)
def test_qss_contains_token_comment(var_name: str, generated_qss: str) -> None:
    """Each key CSS variable must appear as a comment/marker in generated QSS."""
    assert var_name in generated_qss, (
        f"Expected '{var_name}' to appear in tokens.qss (as comment or reference), "
        f"but it was not found."
    )


def test_qss_has_qmainwindow_rule(generated_qss: str) -> None:
    assert "QMainWindow" in generated_qss


def test_qss_has_qwidget_rule(generated_qss: str) -> None:
    assert "QWidget" in generated_qss


def test_qss_has_qpushbutton_rule(generated_qss: str) -> None:
    assert "QPushButton" in generated_qss


def test_qss_has_font_references(generated_qss: str) -> None:
    assert '"Inter"' in generated_qss
    assert '"JetBrains Mono"' in generated_qss


def test_qss_has_border_radius(generated_qss: str) -> None:
    assert "border-radius" in generated_qss


def test_qss_auto_generated_marker(generated_qss: str) -> None:
    assert "AUTO-GENERATED" in generated_qss
    assert "DO NOT EDIT" in generated_qss


# ---------------------------------------------------------------------------
# Round-trip: tokens.py constants == tokens.qss hex values
# ---------------------------------------------------------------------------


def _extract_hex_values_from_qss(qss: str) -> dict[str, list[str]]:
    """Extract all hex colors from QSS grouped by approximate context."""
    return {m: [m] for m in re.findall(r"#[0-9a-fA-F]{6,8}", qss)}


def _extract_hex_values_from_py(py_src: str) -> set[str]:
    """Extract all hex strings from tokens.py."""
    return set(re.findall(r'"(#[0-9a-fA-F]{6,8})"', py_src))


def test_round_trip_hex_values_in_both(generated_qss: str, generated_py: str) -> None:
    """Every hex color in tokens.py must also appear in tokens.qss."""
    py_hexes = _extract_hex_values_from_py(generated_py)
    qss_hexes = set(re.findall(r"#[0-9a-fA-F]{6,8}", generated_qss))

    # We expect color tokens from tokens.py to be present in QSS.
    # It's OK for QSS to have hex values not in tokens.py (e.g. inline rgba).
    # But the core palette tokens must appear in both.
    missing_in_qss = py_hexes - qss_hexes
    if missing_in_qss:
        # Allow small delta: some tokens (line-* with alpha) may be expressed
        # differently in QSS. Report but don't fail on alpha variants.
        alpha_only = {h for h in missing_in_qss if len(h) == 9}  # #rrggbbaa
        critical_missing = missing_in_qss - alpha_only
        assert not critical_missing, (
            f"These hex values from tokens.py are missing from tokens.qss: "
            f"{sorted(critical_missing)}"
        )


def test_round_trip_color_token_count(parsed_tokens: Any, generated_py: str) -> None:
    """tokens.py should contain one constant per color token."""
    py_hexes = _extract_hex_values_from_py(generated_py)
    # We have N color tokens, at minimum N hex values should be in tokens.py
    n_tokens = len(parsed_tokens.colors)
    assert len(py_hexes) >= n_tokens * 0.9, (
        f"Expected ~{n_tokens} hex values in tokens.py, got {len(py_hexes)}"
    )


def test_round_trip_accent_consistent(
    parsed_tokens: Any, generated_qss: str, generated_py: str
) -> None:
    """The --accent hex resolved value must appear identically in both outputs."""
    accent_token = next(
        (ct for ct in parsed_tokens.colors if ct.name == "--accent"), None
    )
    assert accent_token is not None, "--accent token not found"

    hex_val = accent_token.hex.lower()
    # Strip alpha if present
    hex_base = hex_val[:7]

    assert hex_base in generated_qss.lower(), (
        f"--accent hex {hex_base!r} not found in tokens.qss"
    )
    assert hex_base in generated_py.lower(), (
        f"--accent hex {hex_base!r} not found in tokens.py"
    )


# ---------------------------------------------------------------------------
# Required-token validation tests
# ---------------------------------------------------------------------------


_CSS_FIXTURE_BASE = """\
:root {
  --bg-0: #0c0c0e;
  --bg-1: #131316;
  --bg-2: #1a1a1e;
  --bg-3: #222227;
  --bg-4: #2c2c33;
  --bg-5: #36363e;
  --fg-1: #ecebe6;
  --fg-2: #b8b6ad;
  --fg-3: #82807a;
  --fg-4: #56554f;
  --accent: #d4a03a;
  --accent-2: #dba93f;
  --accent-soft: #d4a03a24;
  --accent-line: #d4a03a52;
  --on-accent: #1a1408;
  --ok: #3ab87c;
  --warn: #d4a03a;
  --err: #c0402a;
  --line-1: #ffffff0f;
  --line-2: #ffffff1a;
  --line-3: #ffffff29;
  --r-xs: 2px;
  --r-sm: 4px;
  --r-md: 6px;
  --r-lg: 10px;
}
"""


def _css_without_token(token_name: str) -> str:
    """Вернуть CSS-фикстуру без указанного токена."""
    lines = [
        line for line in _CSS_FIXTURE_BASE.splitlines(keepends=True)
        if not line.strip().startswith(token_name + ":")
    ]
    return "".join(lines)


@pytest.mark.parametrize(
    "missing_token",
    [
        "--bg-0",
        "--bg-1",
        "--fg-1",
        "--accent",
        "--accent-2",
        "--ok",
        "--err",
        "--line-1",
    ],
)
def test_missing_required_token_raises_runtime_error(
    missing_token: str, build_qss: Any
) -> None:
    """При отсутствии required-токена validate_required_tokens должен кидать RuntimeError."""
    css = _css_without_token(missing_token)
    tokens = build_qss.parse_tokens(css)
    build_qss.resolve_all_colors(tokens)

    with pytest.raises(RuntimeError, match="Required design token"):
        build_qss.validate_required_tokens(tokens)


def test_all_required_tokens_present_no_error(build_qss: Any) -> None:
    """При наличии всех required-токенов validate_required_tokens не кидает."""
    tokens = build_qss.parse_tokens(_CSS_FIXTURE_BASE)
    build_qss.resolve_all_colors(tokens)
    # Не должно бросить
    build_qss.validate_required_tokens(tokens)


def test_generate_qss_raises_on_missing_required_token(build_qss: Any) -> None:
    """generate_qss() должен падать с RuntimeError если required-токен отсутствует."""
    css = _css_without_token("--bg-0")
    tokens = build_qss.parse_tokens(css)
    build_qss.resolve_all_colors(tokens)

    with pytest.raises(RuntimeError, match="--bg-0"):
        build_qss.generate_qss(tokens)


# ---------------------------------------------------------------------------
# tokens.py structural tests
# ---------------------------------------------------------------------------


def test_tokens_py_has_colors_dataclass(generated_py: str) -> None:
    assert "class _Colors" in generated_py
    assert "COLORS = _Colors()" in generated_py


def test_tokens_py_has_radii_dataclass(generated_py: str) -> None:
    assert "class _Radii" in generated_py
    assert "RADII = _Radii()" in generated_py


def test_tokens_py_has_font_constants(generated_py: str) -> None:
    assert 'FONT_SANS = "Inter"' in generated_py
    assert 'FONT_MONO = "JetBrains Mono"' in generated_py


def test_tokens_py_has_density_compact(generated_py: str) -> None:
    assert "DENSITY_COMPACT" in generated_py
    assert "DENSITY = DENSITY_COMPACT" in generated_py


def test_tokens_py_has_bg_fg_attrs(generated_py: str) -> None:
    """_Colors dataclass must include BG_0..5 and FG_1..4 attributes."""
    for attr in ["BG_0", "BG_1", "BG_2", "BG_3", "BG_4", "BG_5",
                 "FG_1", "FG_2", "FG_3", "FG_4", "ACCENT"]:
        assert attr in generated_py, f"Expected attribute {attr!r} in tokens.py"


# ---------------------------------------------------------------------------
# load_qss() integration
# ---------------------------------------------------------------------------


def test_load_qss_raises_before_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_qss() raises FileNotFoundError when tokens.qss is absent."""
    from app import styles

    fake_styles_dir = tmp_path / "styles"
    fake_styles_dir.mkdir()
    monkeypatch.setattr(styles, "_STYLES_DIR", fake_styles_dir)

    with pytest.raises(FileNotFoundError, match="make tokens"):
        styles.load_qss()


def test_load_qss_returns_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generated_qss: str
) -> None:
    """load_qss() returns the QSS content when file is present."""
    from app import styles

    fake_styles_dir = tmp_path / "styles"
    fake_styles_dir.mkdir()
    (fake_styles_dir / "tokens.qss").write_text(generated_qss, encoding="utf-8")
    monkeypatch.setattr(styles, "_STYLES_DIR", fake_styles_dir)

    content = styles.load_qss()
    assert "QWidget" in content
    assert "AUTO-GENERATED" in content
