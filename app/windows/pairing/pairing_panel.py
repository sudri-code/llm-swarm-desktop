"""app/windows/pairing/pairing_panel.py — Embedded pairing panel для Settings.

PairingPanel — QWidget с QStackedWidget внутри. Встраивается в SettingsScreen,
не является отдельным окном. Отображает все состояния pairing flow из §5 pairing.md.

Зависит от PairingControllerProtocol (Protocol), не от реальной реализации контроллера.
"""

from __future__ import annotations

import io
import socket
from datetime import UTC, datetime

import qrcode
import qrcode.image.pil as _qrpil  # noqa: F401 — side-effect: registers PilImage
from PySide6.QtCore import QByteArray, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from agent.identity import format_fingerprint
from app.styles.tokens import COLORS, FONT_MONO, RADII
from app.windows.pairing.controller_protocol import (
    PairingControllerProtocol,
    PairingEvent,
    PairingFailed,
    PairingFailureReason,
    PairingPolling,
    PairingStarted,
    PairingSucceeded,
)

# ---------------------------------------------------------------------------
# Маппинг PairingFailureReason → (заголовок, текст, текст primary CTA)
# Точные формулировки из docs/components/pairing.md §5.
# ---------------------------------------------------------------------------

_FAILURE_TEXT: dict[
    PairingFailureReason, tuple[str, str, str]
] = {
    PairingFailureReason.EXPIRED: (
        "Время вышло",
        "Время вышло. Код подтверждения действителен 10 минут — мы не дождались "
        "подтверждения в браузере. Начните привязку заново.",
        "Начать заново",
    ),
    PairingFailureReason.DENIED: (
        "Привязка отклонена",
        "Привязка отклонена. В браузере вы нажали «Не подтверждать». "
        "Если это была ошибка — попробуйте ещё раз.",
        "Попробовать снова",
    ),
    PairingFailureReason.NETWORK: (
        "Не удалось связаться с sudri.ru",
        "Не удалось связаться с sudri.ru. Проверьте интернет-соединение и повторите попытку.",
        "Повторить",
    ),
    PairingFailureReason.RATE_LIMITED: (
        "Не удалось связаться с sudri.ru",
        "Слишком много попыток подряд. Подождите немного и попробуйте снова.",
        "Повторить",
    ),
    PairingFailureReason.INVALID_PUBKEY: (
        "Не удалось подготовить устройство",
        "Не удалось подготовить устройство к привязке. Перезапустите приложение; "
        "если ошибка повторится — отправьте логи в поддержку.",
        "Перезапустить",
    ),
    PairingFailureReason.KEYCHAIN_UNAVAILABLE: (
        "Хранилище паролей недоступно",
        "Системное хранилище паролей недоступно. Без него мы не можем безопасно "
        "сохранить данные привязки. Разблокируйте Связку ключей (macOS) / "
        "войдите в учётную запись (Windows) / запустите сеанс с поддержкой "
        "libsecret (Linux) и повторите.",
        "Повторить",
    ),
    PairingFailureReason.UNKNOWN: (
        "Ошибка привязки",
        "Не удалось завершить привязку. Попробуйте ещё раз или отправьте логи в поддержку.",
        "Повторить",
    ),
    PairingFailureReason.CANCELLED: (
        # CANCELLED — промежуточное состояние без UI (§5), возврат в UNPAIRED
        # Но если вдруг придёт через PairingFailed — показываем нейтральный экран
        "Привязка отменена",
        "Привязка была отменена.",
        "Начать заново",
    ),
}

# Индексы страниц в QStackedWidget
_PAGE_WELCOME = 0
_PAGE_STARTING = 1
_PAGE_AWAITING = 2
_PAGE_SUCCESS = 3
_PAGE_ERROR = 4


def _make_primary_button(text: str, parent: QWidget | None = None) -> QPushButton:
    btn = QPushButton(text, parent)
    btn.setProperty("buttonStyle", "primary")
    return btn


def _make_secondary_button(text: str, parent: QWidget | None = None) -> QPushButton:
    btn = QPushButton(text, parent)
    return btn


def _qr_pixmap(data: str, size: int = 200) -> QPixmap:
    """Конвертировать URL в QPixmap QR-кода (PIL Image → QBuffer → QPixmap).

    Использует qrcode[pil] для генерации PNG, затем PIL для resize и
    QPixmap.loadFromData() для загрузки в Qt.
    """
    # ERROR_CORRECT_M доступен через прямой атрибут или через constants;
    # используем числовое значение M=0 напрямую для обхода отсутствия типов
    qr: qrcode.QRCode = qrcode.QRCode(  # type: ignore[type-arg]
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,  # type: ignore[attr-defined]
        box_size=10,
        border=2,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # make_image с PIL backend возвращает PilImage, поддерживающий .save()
    pil_img = qr.make_image(  # type: ignore[assignment]
        fill_color="white",
        back_color="#1a1a1e",
        image_factory=_qrpil.PilImage,
    )
    # Resize через PIL (PilImage.get_image() возвращает PIL.Image)
    inner = pil_img.get_image()  # type: ignore[union-attr]
    inner = inner.resize((size, size))  # type: ignore[union-attr]

    buf = io.BytesIO()
    inner.save(buf, format="PNG")  # type: ignore[union-attr]
    raw: bytes = buf.getvalue()

    pixmap = QPixmap()
    pixmap.loadFromData(QByteArray(raw))
    return pixmap


# ---------------------------------------------------------------------------
# Отдельные страницы
# ---------------------------------------------------------------------------


class _WelcomePage(QWidget):
    """UNPAIRED: приглашение привязать устройство."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Статус-чип «Не привязано»
        chip_row = QHBoxLayout()
        chip_row.setSpacing(6)
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(
            f"background-color: {COLORS.FG_3}; border-radius: 4px;"
        )
        chip_label = QLabel("Не привязано")
        chip_label.setObjectName("PairingChipLabel")
        chip_label.setProperty("textRole", "tertiary")
        chip_row.addWidget(dot)
        chip_row.addWidget(chip_label)
        chip_row.addStretch()
        layout.addLayout(chip_row)

        body = QLabel(
            "Устройство не привязано. Без привязки нода не сможет "
            "начислять токены на ваш аккаунт sudri.ru."
        )
        body.setObjectName("PairingBodyText")
        body.setWordWrap(True)
        body.setProperty("textRole", "secondary")
        layout.addWidget(body)

        self.btn_pair = _make_primary_button("Привязать устройство")
        self.btn_pair.setObjectName("BtnPair")
        layout.addWidget(self.btn_pair)
        layout.addStretch()


class _StartingPage(QWidget):
    """STARTING: spinner + текст ожидания."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.status_label = QLabel("Готовим привязку…")
        self.status_label.setObjectName("PairingStatusLabel")
        self.status_label.setProperty("textRole", "secondary")
        layout.addWidget(self.status_label)

        self.btn_cancel = _make_secondary_button("Отмена")
        self.btn_cancel.setObjectName("BtnCancelStarting")
        layout.addWidget(self.btn_cancel)
        layout.addStretch()


class _AwaitingPage(QWidget):
    """AWAITING_USER / POLLING: user_code, QR, countdown, действия.

    Dynamic property state="awaiting_user"|"polling" для QSS.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        title = QLabel("Подтвердите привязку в браузере")
        title.setObjectName("PairingAwaitingTitle")
        title.setProperty("role", "h2")
        layout.addWidget(title)

        # user_code — JetBrains Mono ≥28pt, цвет ACCENT, selectable
        self.user_code_label = QLabel("---- ----")
        self.user_code_label.setObjectName("UserCodeLabel")
        self.user_code_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.user_code_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        font_code = QFont(FONT_MONO, 32)
        font_code.setStyleHint(QFont.StyleHint.Monospace)
        self.user_code_label.setFont(font_code)
        self.user_code_label.setStyleSheet(f"color: {COLORS.ACCENT};")
        layout.addWidget(self.user_code_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # QR-код
        self._qr_container = QFrame()
        self._qr_container.setObjectName("QRContainer")
        self._qr_container.setProperty("panelStyle", "card")
        self._qr_container.setStyleSheet(
            f"background-color: {COLORS.BG_2}; padding: 16px; border-radius: {RADII.R_MD}px;"
        )
        qr_layout = QVBoxLayout(self._qr_container)
        qr_layout.setContentsMargins(16, 16, 16, 16)

        self.qr_label = QLabel()
        self.qr_label.setObjectName("QRPixmapLabel")
        self.qr_label.setFixedSize(200, 200)
        self.qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Accessible alt-text через toolTip (QLabel не поддерживает настоящий alt)
        self.qr_label.setToolTip("QR-код для подтверждения в браузере")
        self.qr_label.setAccessibleDescription("QR-код для подтверждения в браузере")
        qr_layout.addWidget(self.qr_label, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._qr_container, alignment=Qt.AlignmentFlag.AlignCenter)

        # Countdown
        self.countdown_label = QLabel("10:00")
        self.countdown_label.setObjectName("CountdownLabel")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font_mono_sm = QFont(FONT_MONO, 14)
        self.countdown_label.setFont(font_mono_sm)
        self.countdown_label.setStyleSheet(f"color: {COLORS.FG_3};")
        layout.addWidget(self.countdown_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Индикатор «Ждём подтверждение…» — показывается при state=polling
        self.polling_indicator = QLabel("Ждём подтверждение…")
        self.polling_indicator.setObjectName("PollingIndicator")
        self.polling_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.polling_indicator.setProperty("textRole", "tertiary")
        self.polling_indicator.setVisible(False)
        layout.addWidget(self.polling_indicator, alignment=Qt.AlignmentFlag.AlignCenter)

        # Кнопки — tab order: «Открыть браузер» → «Скопировать код» → «Отмена» (§11 AC14)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_open_browser = _make_primary_button("Открыть браузер ещё раз")
        self.btn_open_browser.setObjectName("BtnOpenBrowser")

        self.btn_copy_code = _make_secondary_button("Скопировать код")
        self.btn_copy_code.setObjectName("BtnCopyCode")

        self.btn_cancel = _make_secondary_button("Отмена")
        self.btn_cancel.setObjectName("BtnCancelAwaiting")

        btn_row.addWidget(self.btn_open_browser)
        btn_row.addWidget(self.btn_copy_code)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        # Tab order: btn_open_browser → btn_copy_code → btn_cancel
        QWidget.setTabOrder(self.btn_open_browser, self.btn_copy_code)
        QWidget.setTabOrder(self.btn_copy_code, self.btn_cancel)

        layout.addStretch()

        # Внутреннее состояние — заполняется при получении PairingStarted
        self._verification_uri: str = ""
        self._user_code_raw: str = ""

    def set_state(self, state: str) -> None:
        """Переключить dynamic property 'state' для QSS-селекторов."""
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)
        self.polling_indicator.setVisible(state == "polling")

    def set_code(self, user_code: str, verification_uri: str) -> None:
        """Установить user_code и URI для QR и кнопки «Открыть браузер»."""
        self._user_code_raw = user_code
        self._verification_uri = verification_uri
        self.user_code_label.setText(user_code)
        pixmap = _qr_pixmap(verification_uri, 200)
        self.qr_label.setPixmap(pixmap)

    def set_countdown(self, remaining_secs: int) -> None:
        """Обновить countdown-метку и цвет."""
        minutes = remaining_secs // 60
        seconds = remaining_secs % 60
        text = f"{minutes}:{seconds:02d}"
        self.countdown_label.setText(text)
        if remaining_secs < 60:
            self.countdown_label.setStyleSheet(f"color: {COLORS.WARN};")
        else:
            self.countdown_label.setStyleSheet(f"color: {COLORS.FG_3};")

    @property
    def verification_uri(self) -> str:
        return self._verification_uri

    @property
    def user_code_normalized(self) -> str:
        """Нормализованный код для clipboard (без дефиса, uppercase)."""
        return self._user_code_raw.replace("-", "").upper()


class _SuccessPage(QWidget):
    """SUCCESS: подтверждение привязки с fingerprint display-14."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Статус-чип «Привязано»
        chip_row = QHBoxLayout()
        chip_row.setSpacing(6)
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(
            f"background-color: {COLORS.OK}; border-radius: 4px;"
        )
        chip_label = QLabel("Привязано")
        chip_label.setProperty("textRole", "ok")
        chip_row.addWidget(dot)
        chip_row.addWidget(chip_label)
        chip_row.addStretch()
        layout.addLayout(chip_row)

        title = QLabel("Устройство привязано")
        title.setObjectName("PairingSuccessTitle")
        title.setProperty("role", "h2")
        layout.addWidget(title)

        # Метка устройства
        self.device_label_label = QLabel()
        self.device_label_label.setObjectName("PairingDeviceLabel")
        self.device_label_label.setProperty("textRole", "secondary")
        layout.addWidget(self.device_label_label)

        # Идентификатор устройства — display-14 (НИКОГДА raw-12)
        id_row = QHBoxLayout()
        id_row.setSpacing(8)
        id_caption = QLabel("Идентификатор устройства:")
        id_caption.setProperty("textRole", "tertiary")
        self.fingerprint_display_label = QLabel()
        self.fingerprint_display_label.setObjectName("FingerprintDisplayLabel")
        font_fp = QFont(FONT_MONO, 13)
        self.fingerprint_display_label.setFont(font_fp)
        self.fingerprint_display_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        id_row.addWidget(id_caption)
        id_row.addWidget(self.fingerprint_display_label)
        id_row.addStretch()
        layout.addLayout(id_row)

        # Дата привязки
        self.paired_at_label = QLabel()
        self.paired_at_label.setObjectName("PairingDateLabel")
        self.paired_at_label.setProperty("textRole", "tertiary")
        layout.addWidget(self.paired_at_label)

        self.btn_done = _make_primary_button("Готово")
        self.btn_done.setObjectName("BtnDone")
        layout.addWidget(self.btn_done)
        layout.addStretch()

    def populate(
        self,
        device_label: str,
        fingerprint_raw: str,
        paired_at: datetime,
    ) -> None:
        """Заполнить данные успешной привязки.

        Args:
            device_label: метка устройства (после санитизации BFF).
            fingerprint_raw: raw 12-символьный отпечаток; отображается как display-14.
            paired_at: дата привязки (UTC).
        """
        self.device_label_label.setText(device_label)
        # Только display-14, raw-12 никогда не отображается
        self.fingerprint_display_label.setText(format_fingerprint(fingerprint_raw))
        local_dt = paired_at.astimezone()
        self.paired_at_label.setText(
            f"Привязано: {local_dt.strftime('%d.%m.%Y %H:%M')}"
        )


class _ErrorPage(QWidget):
    """Единая страница для всех ERROR_* состояний."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._icon_label = QLabel()
        self._icon_label.setObjectName("ErrorIconLabel")
        layout.addWidget(self._icon_label)

        self._title_label = QLabel()
        self._title_label.setObjectName("ErrorTitleLabel")
        self._title_label.setProperty("role", "h2")
        layout.addWidget(self._title_label)

        self._body_label = QLabel()
        self._body_label.setObjectName("ErrorBodyLabel")
        self._body_label.setWordWrap(True)
        self._body_label.setProperty("textRole", "secondary")
        layout.addWidget(self._body_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_primary = _make_primary_button("Повторить")
        self.btn_primary.setObjectName("BtnErrorPrimary")

        self.btn_secondary = _make_secondary_button("Закрыть")
        self.btn_secondary.setObjectName("BtnErrorSecondary")

        btn_row.addWidget(self.btn_primary)
        btn_row.addWidget(self.btn_secondary)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        layout.addStretch()

        self._current_reason: PairingFailureReason | None = None

    def populate(self, reason: PairingFailureReason, override_message: str | None = None) -> None:
        """Установить контент по типу ошибки.

        Args:
            reason: тип ошибки.
            override_message: если задан — заменяет дефолтный текст (для rate-limit с retry_after).
        """
        self._current_reason = reason
        title, body, cta_text = _FAILURE_TEXT.get(
            reason,
            (
                "Ошибка привязки",
                "Произошла неизвестная ошибка. Попробуйте ещё раз.",
                "Повторить",
            ),
        )
        # ERROR_INVALID — только CTA «Перезапустить», нет secondary «Закрыть»
        if reason == PairingFailureReason.INVALID_PUBKEY:
            self.btn_secondary.setVisible(False)
        else:
            self.btn_secondary.setVisible(True)

        # Иконки через текстовые символы (SVG — в волне 3 с ресурсами)
        icon_map = {
            PairingFailureReason.EXPIRED: "⏱",
            PairingFailureReason.DENIED: "ℹ",
            PairingFailureReason.NETWORK: "✕",
            PairingFailureReason.RATE_LIMITED: "✕",
            PairingFailureReason.INVALID_PUBKEY: "✕",
            PairingFailureReason.KEYCHAIN_UNAVAILABLE: "✕",
            PairingFailureReason.UNKNOWN: "✕",
            PairingFailureReason.CANCELLED: "✕",
        }
        color_map = {
            PairingFailureReason.EXPIRED: COLORS.WARN,
            PairingFailureReason.DENIED: COLORS.INFO,
            PairingFailureReason.NETWORK: COLORS.ERR,
            PairingFailureReason.RATE_LIMITED: COLORS.ERR,
            PairingFailureReason.INVALID_PUBKEY: COLORS.ERR,
            PairingFailureReason.KEYCHAIN_UNAVAILABLE: COLORS.ERR,
            PairingFailureReason.UNKNOWN: COLORS.ERR,
            PairingFailureReason.CANCELLED: COLORS.FG_3,
        }

        icon_char = icon_map.get(reason, "✕")
        icon_color = color_map.get(reason, COLORS.ERR)
        self._icon_label.setText(icon_char)
        self._icon_label.setStyleSheet(f"color: {icon_color}; font-size: 24px;")

        self._title_label.setText(title)
        self._body_label.setText(override_message or body)
        self.btn_primary.setText(cta_text)

    @property
    def current_reason(self) -> PairingFailureReason | None:
        return self._current_reason


# ---------------------------------------------------------------------------
# PairingPanel — главный виджет
# ---------------------------------------------------------------------------


class PairingPanel(QWidget):
    """Pairing panel — встроенный виджет для секции «Привязка к sudri.ru» в Settings.

    Использует QStackedWidget с пятью страницами:
      0 — WelcomePage (UNPAIRED)
      1 — StartingPage (STARTING)
      2 — AwaitingPage (AWAITING_USER / POLLING)
      3 — SuccessPage (SUCCESS)
      4 — ErrorPage (ERROR_*)

    Singleton poll-loop: возврат на Settings во время AWAITING_USER/POLLING
    не перезапускает countdown (§5, §11 AC10).

    Signals:
        pairing_succeeded: эмитируется при получении PairingSucceeded event'а.
            Settings подписывается и перечитывает state.json для обновления
            PAIRED-состояния без перезахода в экран.

    Args:
        controller: объект, реализующий PairingControllerProtocol.
        parent: родительский виджет.
    """

    pairing_succeeded = Signal()
    """Эмитируется при успешной привязке устройства.

    Settings-экран подписывается чтобы перечитать state.json и перейти в PAIRED.
    """

    def __init__(
        self,
        controller: PairingControllerProtocol,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("PairingPanel")
        self._controller = controller

        # Внутреннее состояние
        self._expires_at: datetime | None = None
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.setObjectName("PairingStack")

        self._welcome_page = _WelcomePage()
        self._starting_page = _StartingPage()
        self._awaiting_page = _AwaitingPage()
        self._success_page = _SuccessPage()
        self._error_page = _ErrorPage()

        self._stack.addWidget(self._welcome_page)   # 0
        self._stack.addWidget(self._starting_page)  # 1
        self._stack.addWidget(self._awaiting_page)  # 2
        self._stack.addWidget(self._success_page)   # 3
        self._stack.addWidget(self._error_page)     # 4

        root.addWidget(self._stack)

    def _connect_signals(self) -> None:
        # Подписка на события от controller через bridge
        self._controller.bridge.event_received.connect(self._on_pairing_event)

        # Кнопки WelcomePage
        self._welcome_page.btn_pair.clicked.connect(self._on_pair_clicked)

        # Кнопки StartingPage
        self._starting_page.btn_cancel.clicked.connect(self._on_cancel)

        # Кнопки AwaitingPage
        self._awaiting_page.btn_open_browser.clicked.connect(self._on_open_browser)
        self._awaiting_page.btn_copy_code.clicked.connect(self._on_copy_code)
        self._awaiting_page.btn_cancel.clicked.connect(self._on_cancel)

        # Кнопки SuccessPage
        self._success_page.btn_done.clicked.connect(self._on_success_done)

        # Кнопки ErrorPage
        self._error_page.btn_primary.clicked.connect(self._on_error_primary)
        self._error_page.btn_secondary.clicked.connect(self._on_error_secondary)

    # ------------------------------------------------------------------
    # Slots — события от controller
    # ------------------------------------------------------------------

    def _on_pairing_event(self, event: PairingEvent) -> None:
        """Главный диспетчер событий из pairing controller.

        Один match — единая точка входа для всех событий (§4 pairing.md).
        """
        match event:
            case PairingStarted():
                self._handle_started(event)
            case PairingPolling():
                self._handle_polling(event)
            case PairingSucceeded():
                self._handle_succeeded(event)
            case PairingFailed():
                self._handle_failed(event)

    def _handle_started(self, event: PairingStarted) -> None:
        """Переход в AWAITING_USER: показать коды, запустить countdown."""
        self._expires_at = event.expires_at
        self._awaiting_page.set_code(event.user_code, event.verification_uri_complete)
        self._awaiting_page.set_state("awaiting_user")
        self._update_countdown()
        self._countdown_timer.start()

        # Открыть браузер автоматически при первом старте
        QDesktopServices.openUrl(QUrl(event.verification_uri_complete))

        self._stack.setCurrentIndex(_PAGE_AWAITING)

    def _handle_polling(self, event: PairingPolling) -> None:
        """Переход AWAITING_USER → POLLING при первом attempt."""
        if event.attempt == 1:
            self._awaiting_page.set_state("polling")

    def _handle_succeeded(self, event: PairingSucceeded) -> None:
        """Переход в SUCCESS: отобразить fingerprint display-14."""
        self._countdown_timer.stop()
        paired_at = datetime.now(tz=UTC)
        self._success_page.populate(
            device_label=event.label,
            fingerprint_raw=event.fingerprint_raw,
            paired_at=paired_at,
        )
        self._stack.setCurrentIndex(_PAGE_SUCCESS)
        # Уведомляем Settings чтобы перечитать state.json и перейти в PAIRED
        self.pairing_succeeded.emit()

    def _handle_failed(self, event: PairingFailed) -> None:
        """Переход в ERROR_*: отобразить описание ошибки."""
        self._countdown_timer.stop()

        if event.reason == PairingFailureReason.CANCELLED:
            # CANCELLED — промежуточное состояние без UI, возврат в UNPAIRED (§5)
            self._stack.setCurrentIndex(_PAGE_WELCOME)
            return

        self._error_page.populate(event.reason)
        self._stack.setCurrentIndex(_PAGE_ERROR)

    # ------------------------------------------------------------------
    # Slots — кнопки UI
    # ------------------------------------------------------------------

    def _on_pair_clicked(self) -> None:
        """Кнопка «Привязать устройство»: переход в STARTING + вызов start_pairing."""
        self._stack.setCurrentIndex(_PAGE_STARTING)
        label = socket.gethostname() or "Устройство"
        self._controller.start_pairing(label=label, hardware=None)

    def _on_cancel(self) -> None:
        """Отмена: вызов cancel_pairing, возврат в UNPAIRED."""
        self._countdown_timer.stop()
        self._controller.cancel_pairing()
        self._stack.setCurrentIndex(_PAGE_WELCOME)

    def _on_open_browser(self) -> None:
        """«Открыть браузер ещё раз»: QDesktopServices, без повторного /start."""
        uri = self._awaiting_page.verification_uri
        if uri:
            QDesktopServices.openUrl(QUrl(uri))

    def _on_copy_code(self) -> None:
        """«Скопировать код»: нормализованная форма в clipboard + toast 1.5с."""
        code = self._awaiting_page.user_code_normalized
        QApplication.clipboard().setText(code)
        self._show_toast("Код скопирован")

    def _on_success_done(self) -> None:
        """«Готово»: SUCCESS закрыт — Settings показывает PAIRED state.

        В волне 3 этот сигнал будет проброшен вверх для обновления Settings.
        """
        # Пока просто остаёмся на SuccessPage — Settings перечитает state.json
        pass

    def _on_error_primary(self) -> None:
        """Primary CTA на ErrorPage: retry или restart."""
        reason = self._error_page.current_reason
        if reason == PairingFailureReason.INVALID_PUBKEY:
            # «Перезапустить» — в волне 3 сигнал на main для QApplication.quit()
            QApplication.quit()
        else:
            # «Повторить» / «Начать заново» / «Попробовать снова» → STARTING
            self._stack.setCurrentIndex(_PAGE_STARTING)
            self._controller.retry_pairing()

    def _on_error_secondary(self) -> None:
        """Secondary CTA «Закрыть»: возврат в UNPAIRED."""
        self._stack.setCurrentIndex(_PAGE_WELCOME)

    # ------------------------------------------------------------------
    # Countdown
    # ------------------------------------------------------------------

    def _on_countdown_tick(self) -> None:
        """Каждую секунду пересчитывать remaining; при 0 → ERROR_EXPIRED."""
        self._update_countdown()

    def _update_countdown(self) -> None:
        if self._expires_at is None:
            return
        now = datetime.now(tz=UTC)
        remaining = int((self._expires_at - now).total_seconds())
        if remaining <= 0:
            self._countdown_timer.stop()
            # UI опережает BFF — переходим в ERROR_EXPIRED (§5, §11 AC5)
            self._error_page.populate(PairingFailureReason.EXPIRED)
            self._stack.setCurrentIndex(_PAGE_ERROR)
            return
        self._awaiting_page.set_countdown(remaining)

    # ------------------------------------------------------------------
    # Toast helper
    # ------------------------------------------------------------------

    def _show_toast(self, text: str, duration_ms: int = 1500) -> None:
        """Показать мини-toast поверх виджета на duration_ms миллисекунд."""
        toast = QLabel(text, self)
        toast.setObjectName("PairingToast")
        toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toast.setStyleSheet(
            f"background-color: {COLORS.BG_4}; color: {COLORS.FG_1};"
            f"border-radius: {RADII.R_SM}px; padding: 6px 12px;"
        )
        toast.adjustSize()
        # Центрируем поверх виджета
        pw = self.width()
        ph = self.height()
        tw = toast.width()
        th = toast.height()
        toast.move((pw - tw) // 2, ph - th - 16)
        toast.show()
        QTimer.singleShot(duration_ms, toast.deleteLater)

    # ------------------------------------------------------------------
    # Public accessors (для тестов)
    # ------------------------------------------------------------------

    @property
    def stack(self) -> QStackedWidget:
        return self._stack

    @property
    def welcome_page(self) -> _WelcomePage:
        return self._welcome_page

    @property
    def starting_page(self) -> _StartingPage:
        return self._starting_page

    @property
    def awaiting_page(self) -> _AwaitingPage:
        return self._awaiting_page

    @property
    def success_page(self) -> _SuccessPage:
        return self._success_page

    @property
    def error_page(self) -> _ErrorPage:
        return self._error_page

    @property
    def countdown_timer(self) -> QTimer:
        return self._countdown_timer
