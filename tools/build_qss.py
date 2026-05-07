"""tools/build_qss.py — генератор design-tokens.

Парсит ../llm-swarm-webclient/frontend/src/styles/tokens.css,
конвертирует OKLCH → sRGB hex (через colour-science),
эмитирует:
  app/styles/tokens.qss  — QSS-файл для PySide6 (Qt 6)
  app/styles/tokens.py   — Python-модуль с типизированными константами

Запуск: uv run python tools/build_qss.py
        make tokens
"""

from __future__ import annotations

import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_QSS = REPO_ROOT / "app" / "styles" / "tokens.qss"
OUT_PY = REPO_ROOT / "app" / "styles" / "tokens.py"

_TOKENS_CSS_WEBCLIENT_LIVE = (
    REPO_ROOT.parent
    / "llm-swarm-webclient"
    / "frontend"
    / "src"
    / "styles"
    / "tokens.css"
)
_TOKENS_CSS_VENDOR_SNAPSHOT = REPO_ROOT / "vendor" / "tokens.css"


def _resolve_tokens_source() -> tuple[Path, str]:
    """Locate tokens.css with a two-step fallback.

    Returns (path, source_label) where source_label is one of:
      'webclient-live'     — ../llm-swarm-webclient checked out locally (dev)
      'vendor-snapshot'    — vendor/tokens.css pinned snapshot (CI / no webclient)

    Raises RuntimeError if neither source is available.
    """
    if _TOKENS_CSS_WEBCLIENT_LIVE.exists():
        return _TOKENS_CSS_WEBCLIENT_LIVE, "webclient-live"
    if _TOKENS_CSS_VENDOR_SNAPSHOT.exists():
        return _TOKENS_CSS_VENDOR_SNAPSHOT, "vendor-snapshot"
    raise RuntimeError(
        "tokens.css not found. "
        "Either checkout ../llm-swarm-webclient or run "
        "`make sync-tokens` to populate vendor/tokens.css."
    )


# ---------------------------------------------------------------------------
# Типы
# ---------------------------------------------------------------------------


@dataclass
class ColorToken:
    name: str  # --accent, --bg-0, …
    raw: str  # строка из CSS
    hex: str = ""  # результат конвертации


@dataclass
class ParsedTokens:
    colors: list[ColorToken] = field(default_factory=list)  # type: ignore[arg-type]
    radii: dict[str, str] = field(default_factory=dict)  # type: ignore[arg-type]
    fonts: dict[str, str] = field(default_factory=dict)  # type: ignore[arg-type]
    sizes: dict[str, str] = field(default_factory=dict)  # type: ignore[arg-type]
    density_default: dict[str, str] = field(default_factory=dict)  # type: ignore[arg-type]
    density_compact: dict[str, str] = field(default_factory=dict)  # type: ignore[arg-type]
    density_comfy: dict[str, str] = field(default_factory=dict)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Разбор OKLCH
# ---------------------------------------------------------------------------

_OKLCH_RE = re.compile(
    r"oklch\(\s*"
    r"([\d.]+(?:%)?)\s+"
    r"([\d.]+)\s+"
    r"([\d.]+)"
    r"(?:\s*/\s*([\d.]+))?"  # optional alpha
    r"\s*\)",
    re.IGNORECASE,
)

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{3,8}$")
_RGBA_RE = re.compile(
    r"rgba?\(\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\s*\)"
)


def _oklch_to_hex(raw: str) -> str:
    """Конвертирует строку oklch(…) → #rrggbb.

    При наличии alpha-канала возвращает #rrggbbaa.
    Использует colour-science для точного OKLCH→OKLab→XYZ→sRGB.

    Намеренно использует прямые функции (Oklab_to_XYZ, XYZ_to_sRGB)
    вместо colour.convert(), которая требует networkx.
    """
    import math

    import colour  # colour-science

    m = _OKLCH_RE.search(raw)
    if not m:
        raise ValueError(f"Cannot parse oklch from: {raw!r}")

    l_raw, c_raw, h_raw, a_raw = m.group(1), m.group(2), m.group(3), m.group(4)

    L = float(l_raw.rstrip("%")) / 100.0 if l_raw.endswith("%") else float(l_raw)
    C = float(c_raw)
    H = float(h_raw)

    # OKLCH → OKLab
    H_rad = math.radians(H)
    a_ok = C * math.cos(H_rad)
    b_ok = C * math.sin(H_rad)
    oklab = [L, a_ok, b_ok]

    # OKLab → XYZ D65 (прямая функция, без colour.convert())
    xyz = colour.models.Oklab_to_XYZ(oklab)

    # XYZ → sRGB (включает gamma encoding через apply_cctf_encoding=True)
    rgb_srgb = colour.models.XYZ_to_sRGB(xyz)

    def _clamp(v: float) -> int:
        return max(0, min(255, round(float(v) * 255)))

    r, g, b = _clamp(rgb_srgb[0]), _clamp(rgb_srgb[1]), _clamp(rgb_srgb[2])

    if a_raw is not None:
        a = max(0, min(255, round(float(a_raw) * 255)))
        return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgba_to_hex(raw: str) -> str:
    """rgba(r,g,b,a) → #rrggbbaa или #rrggbb."""
    m = _RGBA_RE.search(raw)
    if not m:
        raise ValueError(f"Cannot parse rgba: {raw!r}")
    r, g, b = int(float(m.group(1))), int(float(m.group(2))), int(float(m.group(3)))
    if m.group(4) is not None:
        a = max(0, min(255, round(float(m.group(4)) * 255)))
        return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
    return f"#{r:02x}{g:02x}{b:02x}"


def resolve_color(name: str, raw: str) -> str:
    """Приводит любое CSS-значение цвета к hex."""
    raw = raw.strip()
    if _HEX_RE.match(raw):
        return raw.lower()
    if "oklch" in raw.lower():
        return _oklch_to_hex(raw)
    if raw.lower().startswith("rgba") or raw.lower().startswith("rgb"):
        return _rgba_to_hex(raw)
    raise ValueError(f"Unsupported color format for {name!r}: {raw!r}")


# ---------------------------------------------------------------------------
# Парсинг tokens.css
# ---------------------------------------------------------------------------

# Переменные, которые считаются цветами
_COLOR_KEYWORDS = {
    "--bg-",
    "--fg-",
    "--accent",
    "--on-accent",
    "--ok",
    "--warn",
    "--err",
    "--info",
    "--m-",
    "--line-",
}

_RADIUS_KEYWORDS = "--r-"
_FONT_KEYWORDS = "--font-"
_SIZE_KEYWORDS = {"--u", "--pad-", "--row-h", "--gap"}


def _is_color_var(name: str) -> bool:
    return any(name.startswith(k) for k in _COLOR_KEYWORDS)


def _is_radius_var(name: str) -> bool:
    return name.startswith(_RADIUS_KEYWORDS)


def _is_font_var(name: str) -> bool:
    return name.startswith(_FONT_KEYWORDS)


def _is_size_var(name: str) -> bool:
    return any(name.startswith(k) for k in _SIZE_KEYWORDS)


_VAR_RE = re.compile(r"(--[\w-]+)\s*:\s*(.+?)\s*;", re.DOTALL)


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def parse_tokens(css: str) -> ParsedTokens:
    tokens = ParsedTokens()
    clean = _strip_comments(css)

    # Найдём блок :root { … }
    root_m = re.search(r":root\s*\{([^}]+)\}", clean, re.DOTALL)
    if not root_m:
        raise ValueError("No :root block found in tokens.css")
    root_block = root_m.group(1)

    for m in _VAR_RE.finditer(root_block):
        name, value = m.group(1).strip(), m.group(2).strip()
        value = re.sub(r"\s+", " ", value)

        if _is_color_var(name):
            tokens.colors.append(ColorToken(name=name, raw=value))
        elif _is_radius_var(name):
            tokens.radii[name] = value
        elif _is_font_var(name):
            tokens.fonts[name] = value
        elif _is_size_var(name):
            tokens.sizes[name] = value
        else:
            # shadow и прочие — кладём как размеры (не используем в QSS напрямую)
            tokens.sizes[name] = value

    # density blocks
    for density_name, attr in [
        ("compact", "density_compact"),
        ("comfy", "density_comfy"),
    ]:
        pat = rf'\[data-density="{density_name}"\]\s*\{{([^}}]+)\}}'
        bm = re.search(pat, clean, re.DOTALL)
        if bm:
            block = bm.group(1)
            d: dict[str, str] = {}
            for vm in _VAR_RE.finditer(block):
                d[vm.group(1).strip()] = vm.group(2).strip()
            setattr(tokens, attr, d)

    return tokens


# ---------------------------------------------------------------------------
# Конвертация цветов
# ---------------------------------------------------------------------------


def resolve_all_colors(tokens: ParsedTokens) -> None:
    """Заполняет .hex для всех ColorToken на месте."""
    errors: list[str] = []
    for ct in tokens.colors:
        try:
            ct.hex = resolve_color(ct.name, ct.raw)
        except Exception as exc:
            errors.append(f"  {ct.name}: {exc}")

    if errors:
        msg = "Color resolution errors:\n" + "\n".join(errors)
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Генерация QSS
# ---------------------------------------------------------------------------

_QSS_HEADER = """\
/*
 * AUTO-GENERATED — DO NOT EDIT MANUALLY
 * Source: ../llm-swarm-webclient/frontend/src/styles/tokens.css
 * Generator: tools/build_qss.py
 * Regenerate: make tokens
 *
 * Design tokens as QSS.  Qt widgets styled here follow the same
 * dark-only warm-amber palette as sudri.ru web client.
 *
 * Colour tokens (CSS var name → resolved sRGB hex):
{color_comments}
 */
"""


def _build_color_comments(tokens: ParsedTokens) -> str:
    lines: list[str] = []
    for ct in tokens.colors:
        lines.append(f" *   {ct.name}: {ct.raw}  =>  {ct.hex}")
    return "\n".join(lines)


def _radius(tokens: ParsedTokens, name: str) -> str:
    v = tokens.radii.get(name, "4px")
    # Qt не понимает 999px для pill; ограничим
    if v == "999px":
        v = "50px"
    return v


# ---------------------------------------------------------------------------
# Required / optional token whitelists
# ---------------------------------------------------------------------------

# Токены без которых QSS не генерируется.
# Если токен исчез из webclient tokens.css — сборка падает явно, без тихого рассинхрона.
_REQUIRED_TOKENS: frozenset[str] = frozenset({
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
    "--accent-line",
    "--on-accent",
    "--ok",
    "--warn",
    "--err",
    "--line-1",
    "--line-2",
    "--line-3",
})

# Токены, для которых допустим fallback (второстепенные, могут отсутствовать в старых версиях CSS).
# Добавляй сюда только осознанно, с комментарием.
_OPTIONAL_TOKEN_DEFAULTS: dict[str, str] = {
    # На случай если webclient не добавил --accent-1 / --border-1 в свой файл (псевдонимы)
    "--accent-1": "",  # fallback разрешён только если совпадает с --accent
    "--border-1": "",  # alias для --line-2 в некоторых контекстах
}


def _require_token(c: dict[str, str], name: str) -> str:
    """Возвращает hex-значение required-токена или кидает RuntimeError."""
    if name not in c or not c[name]:
        raise RuntimeError(
            f"Required design token {name!r} missing in tokens.css; "
            "check webclient sync (../llm-swarm-webclient/frontend/src/styles/tokens.css)"
        )
    return c[name]


def validate_required_tokens(tokens: ParsedTokens) -> None:
    """Проверяет наличие всех required-токенов после парсинга и конвертации.

    Вызывать после resolve_all_colors().
    """
    c = {ct.name: ct.hex for ct in tokens.colors}
    missing = [name for name in sorted(_REQUIRED_TOKENS) if not c.get(name)]
    if missing:
        raise RuntimeError(
            "Required design token(s) missing in tokens.css; check webclient sync: "
            + ", ".join(missing)
        )


def generate_qss(tokens: ParsedTokens) -> str:
    c = {ct.name: ct.hex for ct in tokens.colors}

    # Required tokens — падаем явно если нет
    bg0 = _require_token(c, "--bg-0")
    bg1 = _require_token(c, "--bg-1")
    bg2 = _require_token(c, "--bg-2")
    bg3 = _require_token(c, "--bg-3")
    bg4 = _require_token(c, "--bg-4")
    bg5 = _require_token(c, "--bg-5")

    fg1 = _require_token(c, "--fg-1")
    fg2 = _require_token(c, "--fg-2")
    fg3 = _require_token(c, "--fg-3")
    fg4 = _require_token(c, "--fg-4")

    accent = _require_token(c, "--accent")
    accent2 = _require_token(c, "--accent-2")
    accent_soft = _require_token(c, "--accent-soft")
    accent_line = _require_token(c, "--accent-line")
    on_accent = _require_token(c, "--on-accent")

    ok = _require_token(c, "--ok")
    warn = _require_token(c, "--warn")
    err = _require_token(c, "--err")

    line1 = _require_token(c, "--line-1")
    line2 = _require_token(c, "--line-2")
    line3 = _require_token(c, "--line-3")

    r_xs = _radius(tokens, "--r-xs")
    r_sm = _radius(tokens, "--r-sm")
    r_md = _radius(tokens, "--r-md")
    r_lg = _radius(tokens, "--r-lg")

    # FIXME: font-size hardcoded — нужно перенести в tokens.css webclient'а через
    # cross-repo feature-request (../llm-swarm-webclient) когда понадобится управление
    # размерами шрифта через дизайн-систему. Сейчас нет CSS-переменных для font-size.
    font_size_body = "14px"
    font_size_sm = "12px"

    header = _QSS_HEADER.format(
        color_comments=_build_color_comments(tokens)
    )

    body = f"""\
/* ============================================================
 * Global reset
 * ============================================================ */

QWidget {{
    background-color: {bg1};
    color: {fg1};
    font-family: "Inter";
    font-size: {font_size_body};
    selection-background-color: {accent_soft};
    selection-color: {fg1};
    outline: none;
}}

/* ============================================================
 * Main window / dialog
 * ============================================================ */

QMainWindow {{
    background-color: {bg0};
}}

QDialog {{
    background-color: {bg2};
    border-radius: {r_lg};
}}

/* ============================================================
 * Panels and frames
 * ============================================================ */

QFrame {{
    background-color: transparent;
    border: none;
}}

QFrame[panelStyle="card"] {{
    background-color: {bg2};
    border: 1px solid {line1};
    border-radius: {r_lg};
}}

QFrame[panelStyle="raised"] {{
    background-color: {bg3};
    border: 1px solid {line2};
    border-radius: {r_md};
}}

/* ============================================================
 * Buttons
 * ============================================================ */

QPushButton {{
    background-color: {bg3};
    color: {fg1};
    border: 1px solid {line2};
    border-radius: {r_sm};
    padding: 8px 16px;
    font-size: {font_size_body};
    font-family: "Inter";
    min-height: 30px;
    outline: none;
}}

QPushButton:hover {{
    background-color: {bg4};
    border-color: {line3};
}}

QPushButton:pressed {{
    background-color: {bg5};
}}

QPushButton:disabled {{
    color: {fg4};
    border-color: {line1};
}}

QPushButton:focus {{
    border: 2px solid {accent};
}}

/* Primary / accent button */
QPushButton[buttonStyle="primary"] {{
    background-color: {accent};
    color: {on_accent};
    border: none;
    font-weight: 600;
}}

QPushButton[buttonStyle="primary"]:hover {{
    background-color: {accent2};
}}

QPushButton[buttonStyle="primary"]:pressed {{
    background-color: {accent};
    opacity: 0.9;
}}

/* Destructive button */
QPushButton[buttonStyle="destructive"] {{
    background-color: transparent;
    color: {err};
    border: 1px solid {err};
}}

QPushButton[buttonStyle="destructive"]:hover {{
    background-color: {err};
    color: {fg1};
}}

/* Ghost button */
QPushButton[buttonStyle="ghost"] {{
    background-color: transparent;
    color: {fg2};
    border: none;
}}

QPushButton[buttonStyle="ghost"]:hover {{
    color: {fg1};
    background-color: {bg4};
}}

/* ============================================================
 * Labels
 * ============================================================ */

QLabel {{
    background-color: transparent;
    color: {fg1};
    font-family: "Inter";
}}

QLabel[textRole="secondary"] {{
    color: {fg2};
    font-size: {font_size_sm};
}}

QLabel[textRole="tertiary"] {{
    color: {fg3};
    font-size: {font_size_sm};
}}

QLabel[textRole="disabled"] {{
    color: {fg4};
}}

QLabel[textRole="accent"] {{
    color: {accent};
}}

QLabel[textRole="ok"] {{
    color: {ok};
}}

QLabel[textRole="warn"] {{
    color: {warn};
}}

QLabel[textRole="err"] {{
    color: {err};
}}

QLabel[textRole="code"] {{
    font-family: "JetBrains Mono";
    font-size: {font_size_sm};
    color: {fg2};
}}

/* ============================================================
 * Line edits / text inputs
 * ============================================================ */

QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {bg2};
    color: {fg1};
    border: 1px solid {line2};
    border-radius: {r_sm};
    padding: 6px 10px;
    font-family: "Inter";
    font-size: {font_size_body};
    selection-background-color: {accent_soft};
    selection-color: {fg1};
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 2px solid {accent};
    outline: none;
}}

QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
    color: {fg4};
    border-color: {line1};
}}

/* Mono variant for logs */
QPlainTextEdit[monoFont="true"] {{
    font-family: "JetBrains Mono";
    font-size: {font_size_sm};
    line-height: 1.6;
}}

/* ============================================================
 * Scrollbars
 * ============================================================ */

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {line2};
    border-radius: 4px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background: {line3};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
    height: 0;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {line2};
    border-radius: 4px;
    min-width: 24px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {line3};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
    width: 0;
}}

/* ============================================================
 * Progress bars (chunk download / loading)
 * ============================================================ */

QProgressBar {{
    background-color: {bg3};
    border: 1px solid {line1};
    border-radius: {r_xs};
    height: 6px;
    text-align: center;
    color: transparent;
}}

QProgressBar::chunk {{
    background-color: {accent};
    border-radius: {r_xs};
}}

/* ============================================================
 * Tabs (navigation)
 * ============================================================ */

QTabBar::tab {{
    background-color: transparent;
    color: {fg3};
    padding: 8px 16px;
    border: none;
    border-bottom: 2px solid transparent;
    font-family: "Inter";
    font-size: {font_size_body};
}}

QTabBar::tab:selected {{
    color: {fg1};
    border-bottom: 2px solid {accent};
}}

QTabBar::tab:hover:!selected {{
    color: {fg2};
    background-color: {bg3};
}}

QTabWidget::pane {{
    border: none;
    border-top: 1px solid {line1};
    background-color: {bg1};
}}

/* ============================================================
 * Combo boxes
 * ============================================================ */

QComboBox {{
    background-color: {bg3};
    color: {fg1};
    border: 1px solid {line2};
    border-radius: {r_sm};
    padding: 6px 10px;
    font-family: "Inter";
    font-size: {font_size_body};
    min-height: 30px;
}}

QComboBox:focus {{
    border: 2px solid {accent};
}}

QComboBox:hover {{
    border-color: {line3};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox QAbstractItemView {{
    background-color: {bg3};
    color: {fg1};
    border: 1px solid {line2};
    border-radius: {r_sm};
    selection-background-color: {bg4};
    selection-color: {fg1};
    outline: none;
}}

/* ============================================================
 * Checkboxes / Radio buttons
 * ============================================================ */

QCheckBox {{
    color: {fg1};
    font-family: "Inter";
    font-size: {font_size_body};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {line3};
    border-radius: {r_xs};
    background-color: transparent;
}}

QCheckBox::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}

QCheckBox::indicator:hover {{
    border-color: {accent_line};
}}

QRadioButton {{
    color: {fg1};
    font-family: "Inter";
    font-size: {font_size_body};
    spacing: 8px;
}}

QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {line3};
    border-radius: 8px;
    background-color: transparent;
}}

QRadioButton::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}

/* ============================================================
 * Sliders (throttle, etc.)
 * ============================================================ */

QSlider::groove:horizontal {{
    background: {bg3};
    height: 4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {accent};
    width: 16px;
    height: 16px;
    border-radius: 8px;
    margin: -6px 0;
}}

QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 2px;
}}

/* ============================================================
 * Menu (tray context menu, etc.)
 * ============================================================ */

QMenu {{
    background-color: {bg2};
    color: {fg1};
    border: 1px solid {line2};
    border-radius: {r_md};
    padding: 4px 0;
    font-family: "Inter";
    font-size: {font_size_body};
}}

QMenu::item {{
    padding: 8px 16px;
    border-radius: {r_xs};
    background-color: transparent;
}}

QMenu::item:selected {{
    background-color: {bg4};
    color: {fg1};
}}

QMenu::item:disabled {{
    color: {fg4};
}}

QMenu::separator {{
    height: 1px;
    background-color: {line1};
    margin: 4px 8px;
}}

/* ============================================================
 * Tool tips
 * ============================================================ */

QToolTip {{
    background-color: {bg3};
    color: {fg1};
    border: 1px solid {line2};
    border-radius: {r_xs};
    padding: 4px 8px;
    font-family: "Inter";
    font-size: {font_size_sm};
}}

/* ============================================================
 * Status bar
 * ============================================================ */

QStatusBar {{
    background-color: {bg0};
    color: {fg3};
    border-top: 1px solid {line1};
    font-size: {font_size_sm};
}}

/* ============================================================
 * Splitter
 * ============================================================ */

QSplitter::handle {{
    background-color: {line1};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

/* ============================================================
 * List / tree views
 * ============================================================ */

QListView, QTreeView, QTableView {{
    background-color: {bg1};
    color: {fg1};
    border: 1px solid {line1};
    border-radius: {r_sm};
    alternate-background-color: {bg2};
    outline: none;
    font-family: "Inter";
    font-size: {font_size_body};
}}

QListView::item, QTreeView::item, QTableView::item {{
    padding: 4px 8px;
    border: none;
}}

QListView::item:selected, QTreeView::item:selected, QTableView::item:selected {{
    background-color: {bg4};
    color: {fg1};
}}

QListView::item:hover, QTreeView::item:hover, QTableView::item:hover {{
    background-color: {bg3};
}}

QHeaderView::section {{
    background-color: {bg2};
    color: {fg3};
    border: none;
    border-bottom: 1px solid {line2};
    padding: 4px 8px;
    font-size: {font_size_sm};
}}

/* ============================================================
 * Group boxes (Settings sections)
 * ============================================================ */

QGroupBox {{
    border: 1px solid {line2};
    border-radius: {r_md};
    margin-top: 12px;
    padding: 12px;
    font-family: "Inter";
    font-size: {font_size_sm};
    color: {fg3};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: {fg3};
}}

/* ============================================================
 * Spin boxes
 * ============================================================ */

QSpinBox, QDoubleSpinBox {{
    background-color: {bg2};
    color: {fg1};
    border: 1px solid {line2};
    border-radius: {r_sm};
    padding: 6px 10px;
    font-family: "Inter";
    font-size: {font_size_body};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 2px solid {accent};
}}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: transparent;
    border: none;
    width: 16px;
}}
"""

    return header + body


# ---------------------------------------------------------------------------
# Генерация tokens.py
# ---------------------------------------------------------------------------


def _py_name(css_var: str) -> str:
    """--bg-0 → BG_0, --accent-soft → ACCENT_SOFT."""
    return css_var.lstrip("-").replace("-", "_").upper()


def generate_py(tokens: ParsedTokens) -> str:
    lines: list[str] = [
        '"""app/styles/tokens.py — AUTO-GENERATED, DO NOT EDIT.',
        "",
        "Source: ../llm-swarm-webclient/frontend/src/styles/tokens.css",
        "Generator: tools/build_qss.py",
        "Regenerate: make tokens",
        "",
        "Python constants mirroring design tokens for use in",
        "QPainter / dynamic styles / chart rendering.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# Color tokens (sRGB hex)",
        "# ---------------------------------------------------------------------------",
        "",
        "@dataclass(frozen=True)",
        "class _Colors:",
    ]

    for ct in tokens.colors:
        py_attr = _py_name(ct.name)
        lines.append(f'    {py_attr}: str = "{ct.hex}"  # {ct.name}: {ct.raw}')

    lines += [
        "",
        "",
        "COLORS = _Colors()",
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# Radii (px strings for Qt stylesheet / geometry)",
        "# ---------------------------------------------------------------------------",
        "",
        "@dataclass(frozen=True)",
        "class _Radii:",
    ]

    for name, val in sorted(tokens.radii.items()):
        py_attr = _py_name(name)
        # Числовой вариант (int) для QPainter
        px_val = val.replace("px", "").strip()
        try:
            int_val = int(px_val)
            lines.append(
                f'    {py_attr}: int = {min(int_val, 50)}  # {name}: {val}'
            )
        except ValueError:
            lines.append(f'    {py_attr}: str = "{val}"  # {name}')

    lines += [
        "",
        "",
        "RADII = _Radii()",
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# Font families",
        "# ---------------------------------------------------------------------------",
        "",
        'FONT_SANS = "Inter"',
        'FONT_MONO = "JetBrains Mono"',
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# Font sizes",
        "# ---------------------------------------------------------------------------",
        "",
        "FONT_SIZE_SM = 12",
        "FONT_SIZE_BODY = 14",
        "FONT_SIZE_LG = 16",
        "FONT_SIZE_XL = 20",
        "FONT_SIZE_2XL = 24",
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# Density sizes (default = compact per ADR-0001)",
        "# ---------------------------------------------------------------------------",
        "",
        "@dataclass(frozen=True)",
        "class _Density:",
        "    pad_x: int",
        "    pad_y: int",
        "    row_h: int",
        "    gap: int",
        "",
        "",
    ]

    def _parse_px(v: str) -> int:
        try:
            return int(v.replace("px", "").strip())
        except ValueError:
            return 0

    default_sizes = tokens.sizes
    compact = tokens.density_compact
    comfy = tokens.density_comfy

    def _density_block(name: str, src: dict[str, str]) -> list[str]:
        return [
            f"DENSITY_{name.upper()} = _Density(",
            f"    pad_x={_parse_px(src.get('--pad-x', '12px'))},",
            f"    pad_y={_parse_px(src.get('--pad-y', '8px'))},",
            f"    row_h={_parse_px(src.get('--row-h', '30px'))},",
            f"    gap={_parse_px(src.get('--gap', '8px'))},",
            ")",
        ]

    # Default (from :root)
    lines += [
        "DENSITY_DEFAULT = _Density(",
        f"    pad_x={_parse_px(default_sizes.get('--pad-x', '16px'))},",
        f"    pad_y={_parse_px(default_sizes.get('--pad-y', '12px'))},",
        f"    row_h={_parse_px(default_sizes.get('--row-h', '36px'))},",
        f"    gap={_parse_px(default_sizes.get('--gap', '12px'))},",
        ")",
        "",
    ]
    _default_compact = {"--pad-x": "12px", "--pad-y": "8px", "--row-h": "30px", "--gap": "8px"}
    _default_comfy = {"--pad-x": "20px", "--pad-y": "16px", "--row-h": "42px", "--gap": "16px"}
    lines += _density_block("compact", compact if compact else _default_compact)
    lines.append("")
    lines += _density_block("comfy", comfy if comfy else _default_comfy)
    lines.append("")
    lines += [
        "",
        "# Active density — desktop default is compact (notes.md §3.6, ADR-0001)",
        "DENSITY = DENSITY_COMPACT",
    ]

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# QRC — ресурсы (шрифты)
# ---------------------------------------------------------------------------


def generate_qrc() -> str:
    return textwrap.dedent("""\
        <!DOCTYPE RCC>
        <RCC version="1.0">
          <qresource prefix="/fonts">
            <file alias="Inter-Variable.ttf">fonts/Inter-Variable.ttf</file>
            <file alias="JetBrainsMono-Regular.ttf">fonts/JetBrainsMono-Regular.ttf</file>
            <file alias="JetBrainsMono-Medium.ttf">fonts/JetBrainsMono-Medium.ttf</file>
          </qresource>
        </RCC>
    """)


# ---------------------------------------------------------------------------
# Проверка наличия шрифтов
# ---------------------------------------------------------------------------

_REQUIRED_FONTS = [
    "Inter-Variable.ttf",
    "JetBrainsMono-Regular.ttf",
    "JetBrainsMono-Medium.ttf",
]


def ensure_fonts(fonts_dir: Path) -> None:
    """Требует наличия TTF-файлов в app/resources/fonts/.

    Шрифты должны быть в git-репозитории — никакого runtime-скачивания.
    Inter v4.0 и JetBrains Mono v2.304, лицензия SIL OFL 1.1.
    """
    missing = [f for f in _REQUIRED_FONTS if not (fonts_dir / f).exists()]
    if not missing:
        return

    missing_str = ", ".join(missing)
    raise FileNotFoundError(
        f"Fonts missing in app/resources/fonts/: {missing_str}. "
        "They must be tracked in git as part of repo. "
        "Do NOT download at build time — supply the TTF files directly in app/resources/fonts/ "
        "and commit them. See app/resources/fonts/NOTICE.md for sources."
    )


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        tokens_path, source_label = _resolve_tokens_source()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Reading tokens from: {tokens_path} (source: {source_label})")

    css = tokens_path.read_text(encoding="utf-8")

    print("Parsing CSS tokens...")
    tokens = parse_tokens(css)
    print(f"  Found {len(tokens.colors)} color tokens, {len(tokens.radii)} radii")

    print("Resolving OKLCH -> sRGB hex...")
    resolve_all_colors(tokens)

    print("Validating required tokens...")
    validate_required_tokens(tokens)

    # Вывод для диагностики
    for ct in tokens.colors:
        print(f"  {ct.name:25s}  {ct.raw:45s}  →  {ct.hex}")

    print(f"\nGenerating {OUT_QSS}...")
    OUT_QSS.parent.mkdir(parents=True, exist_ok=True)
    OUT_QSS.write_text(generate_qss(tokens), encoding="utf-8")

    print(f"Generating {OUT_PY}...")
    OUT_PY.write_text(generate_py(tokens), encoding="utf-8")

    # QRC
    qrc_path = REPO_ROOT / "app" / "resources" / "resources.qrc"
    if not qrc_path.exists():
        print(f"Generating {qrc_path}...")
        qrc_path.parent.mkdir(parents=True, exist_ok=True)
        qrc_path.write_text(generate_qrc(), encoding="utf-8")
    else:
        print(f"  {qrc_path} already exists, skipping.")

    # Шрифты — обязаны присутствовать в репо, падаем если нет
    fonts_dir = REPO_ROOT / "app" / "resources" / "fonts"
    ensure_fonts(fonts_dir)

    # pyside6-rcc
    resources_rc = REPO_ROOT / "app" / "resources" / "resources_rc.py"
    if qrc_path.exists():
        print("\nRunning pyside6-rcc...")
        import subprocess

        result = subprocess.run(
            ["pyside6-rcc", str(qrc_path), "-o", str(resources_rc)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR: pyside6-rcc failed: {result.stderr.strip()}", file=sys.stderr)
            return 1
        else:
            print(f"  OK: {resources_rc}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
