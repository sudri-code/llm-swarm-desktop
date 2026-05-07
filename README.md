# llm-swarm-desktop

Десктоп-клиент для раздачи мощностей в p2p-сеть `llm-swarm`. Запускает ноду swarm на домашнем ПК, обслуживает диапазон слоёв модели, начисляет earned tokens на аккаунт [sudri.ru](https://sudri.ru).

## Статус

Stage 1 — каркас GUI + design-token pipeline. Инсталляторы — Stage 6.

## Developer setup

```bash
git clone <repo>
cd llm-swarm-desktop
make install      # uv sync --all-extras + make tokens
make test
make dev          # запуск десктоп-приложения
```

`make install` идемпотентен: при повторном вызове uv и build_qss.py не делают лишней работы.

Для форматирования и статического анализа:

```bash
make lint         # ruff check + pyright
```

### Требования окружения

- Python 3.11+
- uv ≥ 0.4 (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- На **Linux** для запуска тестов нужен X11/Wayland или xvfb (`sudo apt-get install xvfb`).
- На **macOS** GPU-зависимости (PyTorch, bitsandbytes) подтягиваются опционально через `uv sync --all-extras`.

### Генерация design tokens

Артефакты `app/styles/tokens.qss` и `app/styles/tokens.py` генерируются из
`../llm-swarm-webclient/frontend/src/styles/tokens.css` и коммитятся в репо.
Для регенерации после изменения tokens.css:

```bash
make tokens
```

Исходник истины — webclient tokens.css. Не редактировать `tokens.qss` / `tokens.py` вручную.

## Vendor

`vendor/` — снимок `node/` + `shared/` из внутренней codebase sudri.ru (см. ADR-0003).
Не редактировать вручную. Для обновления требуется доступ к `../llm-swarm/`:

```bash
make vendor-sync
```

Актуальный pin фиксируется в `swarm-pin.txt` (коммитить вместе с изменениями в `vendor/`).
Подробности: `docs/decisions/0003-swarm-code-distribution.md`.

## Что это и зачем

`llm-swarm-desktop` — produce-сторона экосистемы swarm. Три компонента вместе:

- **`llm-swarm`** — ядро p2p-сети инференса (tracker + node + client SDK), внутренняя codebase команды sudri.ru. Часть кода ноды (`node/` + `shared/`) распространяется в этом репо как `vendor/` — снимок с фиксированным pin'ом, обновляется через `make vendor-sync`. Подробности в `vendor/NOTICE.md` после первого sync'а.
- **`llm-swarm-webclient`** — веб-клиент на sudri.ru (consume-сторона: чат, баланс, история).
- **`llm-swarm-desktop`** (этот репо) — GUI-приложение для обычных пользователей: иконка в трее, hardware probe, выбор модели, баланс earned tokens, one-click pause/resume.

Десктоп — для тех, кому нужна иконка в трее, а не TOML-конфиг.

## Архитектура

Один Python-процесс (PySide6), три слоя:

1. **GUI** — окна Hardware / Models / Earnings / Logs / Settings, системный трей с цветовым статусом, дизайн-паритет с sudri.ru.
2. **Local agent** — обёртка над нодой swarm. Hardware probe, подбор слоёв, чанк-lifecycle, heartbeat, accounting.
3. **Auth bridge** — device authorization к BFF webclient'а: пользователь логинится в браузере на sudri.ru, привязывает устройство.

Каждое устройство имеет собственный Ed25519-keypair; приватный ключ и device-токен хранятся **только** в OS keychain (macOS Keychain / Windows Credential Manager / Linux libsecret).

## Что десктоп НЕ делает

- Не чат-клиент (consume — через sudri.ru).
- Не CLI-замена (power-users → `python -m node.main` напрямую).
- Не tracker и не relay.
- Не billing / payment. Экономика — contribute-to-use (раздай мощности → получи токены).

## Технологический стек

- **Python 3.11+**
- **PySide6** (Qt 6, LGPL-3.0) — GUI и нативный трей.
- **`keyring`** — OS keychain для секретов.

## Лицензия

MIT — см. [LICENSE](LICENSE). Дистрибутив инсталляторов на Stage 6 дополнительно подчиняется LGPL-3.0 Qt6 (динамическая линковка, offer source).
