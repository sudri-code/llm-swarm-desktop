# llm-swarm-desktop

Десктоп-клиент для раздачи мощностей в p2p-сеть `llm-swarm`. Запускает ноду swarm на домашнем ПК, обслуживает диапазон слоёв модели, начисляет earned tokens на аккаунт [sudri.ru](https://sudri.ru).

## Статус

Stage 0 — архитектура. Кода ещё нет. Релиз и инсталляторы появятся после Stage 6.

## Что это и зачем

`llm-swarm-desktop` — produce-сторона экосистемы swarm. Три компонента вместе:

- **`llm-swarm`** — ядро p2p-сети инференса (tracker + node + client SDK). Часть кода ноды (`node/` + `shared/`) распространяется в этом репо как `vendor/` — снимок с фиксированным pin'ом.
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
