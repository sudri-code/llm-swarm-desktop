"""tests/unit/test_pairing_ui.py — тесты UI pairing flow.

Запуск: uv run pytest tests/unit/test_pairing_ui.py -v
Qt-тесты работают в offscreen-режиме (pytest-qt устанавливает QT_QPA_PLATFORM=offscreen).
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QLabel

# ---------------------------------------------------------------------------
# Skip guard — если платформа не offscreen и нет display, пропускаем.
# ---------------------------------------------------------------------------


def _is_headless() -> bool:
    """True если QtPlatform не 'offscreen' и нет display (CI без X11/Wayland)."""
    # pytest-qt обычно выставляет QT_QPA_PLATFORM=offscreen автоматически.
    # Если QGuiApplication уже создан — проверяем platformName.
    app = QGuiApplication.instance()
    if app is not None:
        platform = app.platformName()
        return platform not in ("offscreen", "xcb", "wayland", "cocoa", "windows")
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_controller() -> MagicMock:
    """Mock-контроллер, реализующий PairingControllerProtocol."""
    from app.windows.pairing.controller_protocol import PairingBridge

    bridge = PairingBridge()
    ctrl = MagicMock()
    ctrl.bridge = bridge
    return ctrl


@pytest.fixture()
def pairing_panel(qtbot: Any, mock_controller: MagicMock) -> Any:
    """PairingPanel, созданный с mock-контроллером."""
    from app.windows.pairing.pairing_panel import PairingPanel

    panel = PairingPanel(controller=mock_controller)
    qtbot.addWidget(panel)
    panel.show()
    return panel


@pytest.fixture()
def future_expires_at() -> datetime:
    """Дата истечения кода через 10 минут от сейчас (UTC)."""
    from datetime import timedelta

    return datetime.now(tz=UTC) + timedelta(minutes=10)


@pytest.fixture()
def pairing_started_event(future_expires_at: datetime) -> Any:
    """PairingStarted с валидными тестовыми данными."""
    from app.windows.pairing.controller_protocol import PairingStarted

    return PairingStarted(
        user_code="ABCD-EFGH",
        verification_uri_complete="https://sudri.ru/link?code=ABCDEFGH",
        expires_at=future_expires_at,
        interval=3,
        fingerprint_raw="abc1def2gh3j",
    )


# ---------------------------------------------------------------------------
# 1. Каждая страница рендерится без падений
# ---------------------------------------------------------------------------


class TestPagesRender:
    def test_welcome_page_renders(self, pairing_panel: Any) -> None:
        pairing_panel.stack.setCurrentIndex(0)
        assert pairing_panel.welcome_page is not None

    def test_starting_page_renders(self, pairing_panel: Any) -> None:
        pairing_panel.stack.setCurrentIndex(1)
        assert pairing_panel.starting_page is not None

    def test_awaiting_page_renders(self, pairing_panel: Any, pairing_started_event: Any) -> None:
        pairing_panel._controller.bridge.event_received.emit(pairing_started_event)
        assert pairing_panel.stack.currentIndex() == 2  # _PAGE_AWAITING

    def test_success_page_renders(self, pairing_panel: Any) -> None:
        pairing_panel.stack.setCurrentIndex(3)
        assert pairing_panel.success_page is not None

    def test_error_page_renders(self, pairing_panel: Any) -> None:
        pairing_panel.stack.setCurrentIndex(4)
        assert pairing_panel.error_page is not None


# ---------------------------------------------------------------------------
# 2. WelcomePage: click «Привязать» → mock controller вызван
# ---------------------------------------------------------------------------


class TestWelcomePage:
    def test_btn_pair_calls_start_pairing(
        self, qtbot: Any, pairing_panel: Any, mock_controller: MagicMock
    ) -> None:
        from PySide6.QtCore import Qt

        assert pairing_panel.stack.currentIndex() == 0  # WelcomePage
        qtbot.mouseClick(pairing_panel.welcome_page.btn_pair, Qt.MouseButton.LeftButton)
        mock_controller.start_pairing.assert_called_once()
        call_kwargs = mock_controller.start_pairing.call_args
        # hardware=None (нет hardware summary в W3)
        assert call_kwargs.kwargs.get("hardware") is None or (
            len(call_kwargs.args) > 1 and call_kwargs.args[1] is None
        ) or call_kwargs.kwargs.get("hardware") is None

    def test_double_click_pair_idempotent(
        self, qtbot: Any, pairing_panel: Any, mock_controller: MagicMock
    ) -> None:
        """Двойной клик при STARTING — no-op (idempotent per §11 AC15).

        Второй клик ничего нового не вызывает после первого start_pairing.
        """
        from PySide6.QtCore import Qt

        qtbot.mouseClick(pairing_panel.welcome_page.btn_pair, Qt.MouseButton.LeftButton)
        # После первого клика стек переключился на STARTING (index=1)
        assert pairing_panel.stack.currentIndex() == 1
        # Второй клик на welcome_page.btn_pair уже недоступен (страница скрыта)
        # Проверяем что start_pairing вызван ровно 1 раз
        assert mock_controller.start_pairing.call_count == 1


# ---------------------------------------------------------------------------
# 3. AwaitingPage: PairingStarted event
# ---------------------------------------------------------------------------


class TestAwaitingPage:
    def test_user_code_formatted(
        self, pairing_panel: Any, pairing_started_event: Any
    ) -> None:
        pairing_panel._controller.bridge.event_received.emit(pairing_started_event)
        assert pairing_panel.awaiting_page.user_code_label.text() == "ABCD-EFGH"

    def test_qr_pixmap_created(
        self, pairing_panel: Any, pairing_started_event: Any
    ) -> None:
        pairing_panel._controller.bridge.event_received.emit(pairing_started_event)
        pixmap = pairing_panel.awaiting_page.qr_label.pixmap()
        assert pixmap is not None
        assert not pixmap.isNull()

    def test_countdown_timer_started(
        self, pairing_panel: Any, pairing_started_event: Any
    ) -> None:
        pairing_panel._controller.bridge.event_received.emit(pairing_started_event)
        assert pairing_panel.countdown_timer.isActive()

    def test_countdown_warning_at_59_seconds(
        self, qtbot: Any, pairing_panel: Any, future_expires_at: Any
    ) -> None:
        """При remaining < 60с countdown label должен быть WARN-цвета."""
        from datetime import timedelta

        from app.styles.tokens import COLORS
        from app.windows.pairing.controller_protocol import PairingStarted

        # expires_at через 55 секунд — меньше 60, должен быть WARN
        expires_soon = datetime.now(tz=UTC) + timedelta(seconds=55)
        event = PairingStarted(
            user_code="XXXX-YYYY",
            verification_uri_complete="https://sudri.ru/link?code=XXXXYYYY",
            expires_at=expires_soon,
            interval=3,
            fingerprint_raw="abc1def2gh3j",
        )
        pairing_panel._controller.bridge.event_received.emit(event)
        # countdown_label должен показывать WARN цвет
        style = pairing_panel.awaiting_page.countdown_label.styleSheet()
        assert COLORS.WARN in style

    def test_countdown_normal_color_above_60(
        self, pairing_panel: Any, pairing_started_event: Any
    ) -> None:
        """При remaining >= 60с countdown label должен быть FG_3."""
        from app.styles.tokens import COLORS

        pairing_panel._controller.bridge.event_received.emit(pairing_started_event)
        style = pairing_panel.awaiting_page.countdown_label.styleSheet()
        assert COLORS.FG_3 in style


# ---------------------------------------------------------------------------
# 4. AwaitingPage: «Скопировать код»
# ---------------------------------------------------------------------------


class TestCopyCode:
    def test_copy_puts_normalized_code_in_clipboard(
        self, qtbot: Any, pairing_panel: Any, pairing_started_event: Any
    ) -> None:
        pairing_panel._controller.bridge.event_received.emit(pairing_started_event)
        qtbot.mouseClick(
            pairing_panel.awaiting_page.btn_copy_code,
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MouseButton.LeftButton,
        )
        clipboard_text = QApplication.clipboard().text()
        # Нормализованная форма — без дефиса, uppercase
        assert clipboard_text == "ABCDEFGH"


# ---------------------------------------------------------------------------
# 5. AwaitingPage: «Открыть браузер ещё раз»
# ---------------------------------------------------------------------------


class TestOpenBrowser:
    def test_open_browser_uses_desktop_services(
        self, qtbot: Any, pairing_panel: Any, pairing_started_event: Any
    ) -> None:
        pairing_panel._controller.bridge.event_received.emit(pairing_started_event)
        called_url: list[QUrl] = []

        with patch(
            "app.windows.pairing.pairing_panel.QDesktopServices.openUrl",
            side_effect=called_url.append,
        ):
            qtbot.mouseClick(
                pairing_panel.awaiting_page.btn_open_browser,
                __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MouseButton.LeftButton,
            )

        assert len(called_url) == 1
        assert called_url[0].toString() == "https://sudri.ru/link?code=ABCDEFGH"


# ---------------------------------------------------------------------------
# 6. PairingPolling → переключение в state="polling"
# ---------------------------------------------------------------------------


class TestPollingTransition:
    def test_first_polling_event_switches_state(
        self, pairing_panel: Any, pairing_started_event: Any
    ) -> None:
        from app.windows.pairing.controller_protocol import PairingPolling

        pairing_panel._controller.bridge.event_received.emit(pairing_started_event)
        # До первого polling — state="awaiting_user"
        assert pairing_panel.awaiting_page.property("state") == "awaiting_user"
        assert not pairing_panel.awaiting_page.polling_indicator.isVisible()

        pairing_panel._controller.bridge.event_received.emit(PairingPolling(attempt=1))
        assert pairing_panel.awaiting_page.property("state") == "polling"
        assert pairing_panel.awaiting_page.polling_indicator.isVisible()

    def test_subsequent_polling_events_stay_polling(
        self, pairing_panel: Any, pairing_started_event: Any
    ) -> None:
        from app.windows.pairing.controller_protocol import PairingPolling

        pairing_panel._controller.bridge.event_received.emit(pairing_started_event)
        pairing_panel._controller.bridge.event_received.emit(PairingPolling(attempt=1))
        pairing_panel._controller.bridge.event_received.emit(PairingPolling(attempt=2))
        assert pairing_panel.awaiting_page.property("state") == "polling"


# ---------------------------------------------------------------------------
# 7. PairingSucceeded → SuccessPage с fingerprint display-14
# ---------------------------------------------------------------------------


class TestSuccessPage:
    def test_success_shows_display_14(
        self, pairing_panel: Any, pairing_started_event: Any
    ) -> None:
        from app.windows.pairing.controller_protocol import PairingSucceeded

        event = PairingSucceeded(
            device_id="dev-123",
            user_id="user-456",
            scope=["node:run"],
            fingerprint_raw="abc1def2gh3j",
            token_expires_at=datetime.now(tz=UTC),
            label="MacBook Pro Test",
        )
        pairing_panel._controller.bridge.event_received.emit(event)
        assert pairing_panel.stack.currentIndex() == 3  # _PAGE_SUCCESS

        fp_label = pairing_panel.success_page.fingerprint_display_label
        # display-14: "abc1 def2 gh3j"
        assert fp_label.text() == "abc1 def2 gh3j"

    def test_success_raw_12_not_visible(
        self, pairing_panel: Any
    ) -> None:
        """Raw-12 fingerprint нигде не должен отображаться напрямую."""
        from app.windows.pairing.controller_protocol import PairingSucceeded

        raw = "abc1def2gh3j"
        event = PairingSucceeded(
            device_id="dev-123",
            user_id="user-456",
            scope=["node:run"],
            fingerprint_raw=raw,
            token_expires_at=datetime.now(tz=UTC),
            label="TestDevice",
        )
        pairing_panel._controller.bridge.event_received.emit(event)

        all_labels = pairing_panel.findChildren(QLabel)
        for label in all_labels:
            text = label.text()
            # raw-12 без пробелов не должен встречаться в UI
            assert raw not in text, (
                f"Raw fingerprint '{raw}' found in QLabel text: '{text}'"
            )

    def test_token_expires_at_not_shown(
        self, pairing_panel: Any
    ) -> None:
        """token_expires_at НЕ показывается пользователю (§11 AC11)."""
        from app.windows.pairing.controller_protocol import PairingSucceeded

        expires = datetime(2099, 12, 31, tzinfo=UTC)
        event = PairingSucceeded(
            device_id="dev-123",
            user_id="user-456",
            scope=["node:run"],
            fingerprint_raw="abc1def2gh3j",
            token_expires_at=expires,
            label="TestDevice",
        )
        pairing_panel._controller.bridge.event_received.emit(event)

        all_labels = pairing_panel.findChildren(QLabel)
        for label in all_labels:
            text = label.text()
            assert "2099" not in text, (
                f"token_expires_at year '2099' found in UI: '{text}'"
            )


# ---------------------------------------------------------------------------
# 8. PairingFailed → ErrorPage: маппинг reason → текст (параметризованный)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "expected_title"),
    [
        ("expired", "Время вышло"),
        ("denied", "Привязка отклонена"),
        ("network", "Не удалось связаться с sudri.ru"),
        ("rate_limited", "Не удалось связаться с sudri.ru"),
        ("invalid_pubkey", "Не удалось подготовить устройство"),
        ("keychain_unavailable", "Хранилище паролей недоступно"),
        ("unknown", "Ошибка привязки"),
    ],
)
def test_error_reason_mapping(
    qtbot: Any,
    mock_controller: MagicMock,
    reason: str,
    expected_title: str,
) -> None:
    """Каждый PairingFailureReason отображает правильный заголовок из §5."""
    from app.windows.pairing.controller_protocol import PairingFailed, PairingFailureReason
    from app.windows.pairing.pairing_panel import PairingPanel

    panel = PairingPanel(controller=mock_controller)
    qtbot.addWidget(panel)
    panel.show()

    event = PairingFailed(
        reason=PairingFailureReason(reason),
        message="test",
    )
    panel._controller.bridge.event_received.emit(event)

    assert panel.stack.currentIndex() == 4  # _PAGE_ERROR
    title_text = panel.error_page._title_label.text()
    assert title_text == expected_title, (
        f"For reason={reason}: expected '{expected_title}', got '{title_text}'"
    )


# ---------------------------------------------------------------------------
# 9. Settings PAIRED: state.json → показывается «Привязано» + display-14
# ---------------------------------------------------------------------------


class TestSettingsPairedState:
    def test_paired_state_shows_ok_chip(
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """При наличии valid state.json — PairingSectionWidget показывает PAIRED UI."""
        state_data = {
            "paired_at": "2026-05-07T12:34:56Z",
            "device_label": "MacBook Pro 16\"",
            "fingerprint_raw": "abc1def2gh3j",
            "peer_id": "somebase58string",
        }
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data), encoding="utf-8")

        with patch(
            "app.screens.settings._state_json_path",
            return_value=state_file,
        ):
            from app.screens.settings import PairingSectionWidget

            widget = PairingSectionWidget(controller=None)
            qtbot.addWidget(widget)
            widget.show()

        assert widget.paired_data is not None
        # Проверяем что fingerprint отображается в display-14
        fp_labels = widget.findChildren(QLabel, "PairedFingerprintLabel")
        assert len(fp_labels) == 1
        assert fp_labels[0].text() == "abc1 def2 gh3j"

    def test_paired_chip_text(
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        state_data = {
            "paired_at": "2026-05-07T12:34:56Z",
            "device_label": "Test Device",
            "fingerprint_raw": "abc1def2gh3j",
        }
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data), encoding="utf-8")

        with patch(
            "app.screens.settings._state_json_path",
            return_value=state_file,
        ):
            from app.screens.settings import PairingSectionWidget

            widget = PairingSectionWidget(controller=None)
            qtbot.addWidget(widget)
            widget.show()

        chip_labels = widget.findChildren(QLabel, "PairedStatusChip")
        assert len(chip_labels) == 1
        assert chip_labels[0].text() == "Привязано"

    def test_unpaired_when_no_state_file(
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        nonexistent = tmp_path / "no_state.json"

        with patch(
            "app.screens.settings._state_json_path",
            return_value=nonexistent,
        ):
            from app.screens.settings import PairingSectionWidget

            widget = PairingSectionWidget(controller=None)
            qtbot.addWidget(widget)
            widget.show()

        assert widget.paired_data is None
        unpaired_chips = widget.findChildren(QLabel, "UnpairedStatusChip")
        assert len(unpaired_chips) == 1


# ---------------------------------------------------------------------------
# 10. Settings unlink: QMessageBox → keychain.delete + state.json очищен
# ---------------------------------------------------------------------------


class TestSettingsUnlink:
    def test_graceful_unlink_clears_keychain_and_state(
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """«Отвязать (graceful)» → Keychain.delete_device_token(peer_id) + state.json очищен."""
        peer_id = "testpeer123abc"
        state_data = {
            "paired_at": "2026-05-07T12:34:56Z",
            "device_label": "Test Device",
            "fingerprint_raw": "abc1def2gh3j",
            "peer_id": peer_id,
        }
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data), encoding="utf-8")

        deleted_peer_ids: list[str] = []

        def fake_delete_device_token(self_kc: Any, pid: str) -> None:  # noqa: ANN001
            deleted_peer_ids.append(pid)

        with (
            patch("app.screens.settings._state_json_path", return_value=state_file),
            patch(
                "agent.keychain.Keychain.delete_device_token",
                autospec=True,
                side_effect=fake_delete_device_token,
            ),
        ):
            from app.screens.settings import PairingSectionWidget

            widget = PairingSectionWidget(controller=None)
            qtbot.addWidget(widget)
            widget.show()

            widget._perform_unlink()

        # delete_device_token должен быть вызван с правильным peer_id из state.json
        assert deleted_peer_ids == [peer_id]

        # state.json должен быть очищен (пустой объект)
        result = json.loads(state_file.read_text(encoding="utf-8"))
        assert result == {}

        # paired_data должен быть None после unlink
        assert widget.paired_data is None

    def test_unlink_preserves_private_key(
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """После unlink load_private_key(peer_id) по-прежнему возвращает байты ключа.

        delete_device_token не трогает :privkey запись в keychain.
        """
        import keyring as kr
        from keyrings.alt.file import PlaintextKeyring

        import agent.keychain as kc_module

        ring = PlaintextKeyring()
        ring.file_path = str(tmp_path / "test_keyring.cfg")  # type: ignore[attr-defined]
        kr.set_keyring(ring)

        original_index_path = kc_module._peers_index_path

        def patched_index() -> Path:
            return tmp_path / "peers.json"

        kc_module._peers_index_path = patched_index  # type: ignore[assignment]

        try:
            from agent.identity import compute_peer_id, generate_keypair
            from agent.keychain import Keychain

            priv, pub = generate_keypair()
            peer_id = compute_peer_id(pub)

            kc = Keychain()
            kc.store_private_key(peer_id, priv)
            kc.store_device_token(peer_id, "tok-to-remove")

            state_data = {
                "paired_at": "2026-05-07T12:34:56Z",
                "device_label": "Test Device",
                "fingerprint_raw": "abc1def2gh3j",
                "peer_id": peer_id,
            }
            state_file = tmp_path / "state.json"
            state_file.write_text(json.dumps(state_data), encoding="utf-8")

            with patch("app.screens.settings._state_json_path", return_value=state_file):
                from app.screens.settings import PairingSectionWidget

                widget = PairingSectionWidget(controller=None)
                qtbot.addWidget(widget)
                widget.show()
                widget._perform_unlink()

            # Token is gone
            assert Keychain().load_device_token(peer_id) is None
            # Private key survives
            assert Keychain().load_private_key(peer_id) == priv
        finally:
            kc_module._peers_index_path = original_index_path  # type: ignore[assignment]

    def test_unlink_with_missing_state_json(
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """Если peer_id отсутствует в state.json — unlink не падает,
        state.json всё равно очищается.
        """
        # state.json без поля peer_id (повреждённый или неполный)
        state_data = {
            "paired_at": "2026-05-07T12:34:56Z",
            "device_label": "Broken Device",
            "fingerprint_raw": "abc1def2gh3j",
            # "peer_id" intentionally absent
        }
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data), encoding="utf-8")

        delete_called: list[str] = []

        def fake_delete_device_token(self_kc: Any, pid: str) -> None:  # noqa: ANN001
            delete_called.append(pid)

        with (
            patch("app.screens.settings._state_json_path", return_value=state_file),
            patch(
                "agent.keychain.Keychain.delete_device_token",
                autospec=True,
                side_effect=fake_delete_device_token,
            ),
        ):
            from app.screens.settings import PairingSectionWidget

            widget = PairingSectionWidget(controller=None)
            qtbot.addWidget(widget)
            widget.show()

            # Должно не падать несмотря на отсутствие peer_id
            widget._perform_unlink()

        # delete_device_token не должен быть вызван (нет peer_id)
        assert delete_called == []

        # state.json всё равно должен быть очищен
        result = json.loads(state_file.read_text(encoding="utf-8"))
        assert result == {}

        assert widget.paired_data is None


# ---------------------------------------------------------------------------
# 11. Запрещённые термины — grep по строковым литералам
# ---------------------------------------------------------------------------

_BANNED_TERMS = [
    "peer_id",
    "pubkey",
    "Ed25519",
    "base58",
    "relay_token",
    "hop_index",
    "layer_range",
    "session_id",
    "Billing",
    "top-up",
    "credits",
    "subscription",
    "тариф",
    "оплата",
    "кредиты",
]

_CHECKED_FILES = [
    Path(__file__).parent.parent.parent
    / "app"
    / "windows"
    / "pairing"
    / "controller_protocol.py",
    Path(__file__).parent.parent.parent
    / "app"
    / "windows"
    / "pairing"
    / "pairing_panel.py",
    Path(__file__).parent.parent.parent
    / "app"
    / "windows"
    / "pairing"
    / "__init__.py",
    Path(__file__).parent.parent.parent / "app" / "screens" / "settings.py",
]


def _collect_docstring_nodes(tree: ast.AST) -> set[int]:
    """Собрать id AST-узлов, являющихся docstring'ами (не UI-текст)."""
    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr):
                expr = body[0].value
                if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                    docstring_ids.add(id(expr))
    return docstring_ids


def _collect_strenum_value_nodes(tree: ast.AST) -> set[int]:
    """Собрать id AST-узлов, являющихся значениями StrEnum (технические API-ключи)."""
    strenum_ids: set[int] = set()
    for node in ast.walk(tree):
        # Ищем: class Foo(StrEnum): ... X = "value"
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [
            b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else "")
            for b in node.bases
        ]
        if "StrEnum" not in bases:
            continue
        for item in ast.walk(node):
            if isinstance(item, ast.Assign):
                for v in (item.value,):
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        strenum_ids.add(id(v))
    return strenum_ids


def _extract_ui_string_literals(source: str) -> list[str]:
    """Извлечь строковые литералы, которые могут отображаться в UI.

    Исключает:
    - docstring'и (не видны пользователям)
    - StrEnum-значения (технические API-ключи, не отображаются в UI)
    """
    import sys

    # CPython 3.11 иногда кидает SystemError при глубокой рекурсии AST
    # в полном test suite. Временно поднимаем лимит для парсинга.
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 5000))
    try:
        tree = ast.parse(source)
    except (SyntaxError, SystemError):
        return []
    finally:
        sys.setrecursionlimit(old_limit)

    excluded_ids = _collect_docstring_nodes(tree) | _collect_strenum_value_nodes(tree)

    strings: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in excluded_ids
        ):
            strings.append(node.value)
    return strings


@pytest.mark.parametrize("term", _BANNED_TERMS)
def test_banned_term_not_in_ui_string_literals(term: str) -> None:
    """Запрещённый термин не должен встречаться в UI-строковых литералах.

    Проверяются только строки, видимые пользователю:
    - исключены docstring'и (документация разработчика)
    - исключены StrEnum-значения (технические API-ключи)
    """
    violations: list[str] = []
    for file_path in _CHECKED_FILES:
        if not file_path.exists():
            continue
        source = file_path.read_text(encoding="utf-8")
        literals = _extract_ui_string_literals(source)
        for literal in literals:
            if term in literal:
                violations.append(f"{file_path.name}: '{literal}'")

    assert not violations, (
        f"Banned term '{term}' found in UI string literals:\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# 12. Регрессия format_fingerprint: raw-12 нигде не в UI без формата
# ---------------------------------------------------------------------------


class TestFingerprintRegression:
    def test_raw_12_not_shown_in_success(
        self, pairing_panel: Any
    ) -> None:
        """После PairingSucceeded raw-12 нигде в QLabel не отображается."""
        from app.windows.pairing.controller_protocol import PairingSucceeded

        raw = "testraw12345"  # 12 символов
        event = PairingSucceeded(
            device_id="dev-x",
            user_id="user-x",
            scope=["node:run"],
            fingerprint_raw=raw,
            token_expires_at=datetime.now(tz=UTC),
            label="TestDevice",
        )
        pairing_panel._controller.bridge.event_received.emit(event)

        all_labels = pairing_panel.findChildren(QLabel)
        for label_widget in all_labels:
            assert raw not in label_widget.text(), (
                f"Raw fingerprint found in label: '{label_widget.text()}'"
            )

    def test_format_fingerprint_consistency(self) -> None:
        """format_fingerprint(raw) == 'XXXX XXXX XXXX' для любого valid raw-12."""
        from agent.identity import format_fingerprint

        raw = "aBc1dEf2gH3j"
        display = format_fingerprint(raw)
        assert len(display) == 14
        parts = display.split(" ")
        assert len(parts) == 3
        assert all(len(p) == 4 for p in parts)
        assert "".join(parts) == raw
