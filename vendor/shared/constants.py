# Значения из §4.4 спеки (heartbeat) и §7.1 (spot-check / верификация).
# Менять только синхронно с обновлением спеки.

# Heartbeat (§4.4)
HEARTBEAT_INTERVAL_SEC: int = 15  # нода шлёт heartbeat каждые 15 сек
HEARTBEAT_TIMEOUT_SEC: int = 30  # трекер считает ноду offline если нет heartbeat 30 сек

# Spot-check (§4.5, §7.1)
EPSILON: float = 0.01  # допуск относительной L2-нормы для spot-check
SPOT_CHECK_RATE: float = 0.05  # 1/20 запросов — golden sample
SPOT_CHECK_EPSILON: float = 0.01  # алиас EPSILON — явное имя для spot_check модуля
GOLDEN_CACHE_SIZE: int = 50  # LRU-размер кэша golden samples (CLAUDE.md инвариант)
GOLDEN_CACHE_TTL_SEC: int = 86400  # 24 ч — ротация golden samples (CLAUDE.md инвариант)

# Карантин (§3.3, §3.5)
STRIKE_THRESHOLD: int = 3  # страйков за 24 ч → карантин
STRIKE_WINDOW_HOURS: int = 24  # окно подсчёта страйков
QUARANTINE_DURATION_HOURS: int = 24  # длительность карантина

# Репутация (§3.5)
MIN_RATIO: float = 0.1  # нода с ratio < 0.1 исключается из маршрутов
MAX_RATIO_BONUS: float = 2.0  # cap в формуле ratio_bonus

# Веса → токены (§5.2)
WEIGHT_GB_TO_TOKENS: int = 10_000  # 1 ГБ весов = 10K токенов

# Прочее
KV_CACHE_TTL_SEC: int = 300  # 5 минут неактивности → очистка KV-cache
RESOURCE_MONITOR_INTERVAL_SEC: int = 5  # период опроса ресурсов (§4.6)
NAT_STEP_TIMEOUT_SEC: int = 2  # таймаут каждого шага NAT traversal (§4.4)

# Транспортные лимиты Stage 3 (ADR-0019, §4.4)
# Максимальный размер одного ForwardEnvelope (включая activations_blob).
# Расчёт: hidden 8192 × seq 4096 × INT8 + scales fp16 ≈ 32 MiB.
MAX_ENVELOPE_SIZE_BYTES: int = 32 * 1024 * 1024  # 32 MiB

# Максимальная длина цепочки нод (ADR-0019).
# 70B Llama укладывается в ~10 хопов; 16 даёт запас.
MAX_CHAIN_LENGTH: int = 16

# Максимальное число одновременно in-flight envelope в одной сессии (ADR-0019).
MAX_INFLIGHT_ENVELOPES_PER_SESSION: int = 4

# Backpressure timeout: если получатель не читает дольше N секунд — закрыть сессию (ADR-0019).
BACKPRESSURE_TIMEOUT_SEC: int = 5

# Таймаут forward-хопа (§6.2, ADR-0019).
FORWARD_TIMEOUT_SEC: int = 3

# QUIC dial budget (§4.4, ADR-0019): полный бюджет на установку QUIC-соединения.
QUIC_DIAL_TIMEOUT_SEC: int = 8

# Relay-лимиты (ADR-0014, §3.2)
# Throttle per session: token bucket, backpressure (не drop).
RELAY_MAX_BYTES_PER_SEC: int = 5 * 1024 * 1024  # 5 MiB/s

# Hard cap на сессию: при превышении — WS close 1009.
RELAY_MAX_BYTES_PER_SESSION: int = 1 * 1024**3  # 1 GiB

# Максимальное число одновременных relay-сессий на ноду.
RELAY_MAX_SESSIONS_PER_NODE: int = 4

# Idle timeout: если нет байт ни в одну сторону — close 1000.
RELAY_IDLE_TIMEOUT_SEC: int = 60

# Pair timeout: если второй пир не подключился — close 1011.
RELAY_PAIR_TIMEOUT_SEC: int = 10

# STUN серверы по умолчанию (ADR-0015, §4.4)
STUN_SERVERS_DEFAULT: list[str] = [
    "stun.l.google.com:19302",
    "stun.cloudflare.com:3478",
]

# Таймаут одного STUN-запроса (ADR-0015).
STUN_PROBE_TIMEOUT_SEC: float = 2.0

# Resource Monitor (ADR-0038, §4.6)
MONITOR_INTERVAL_SEC: int = 5  # Период опроса ресурсов, секунды
GPU_TEMP_LIMIT_C: float = 85.0  # Порог температуры GPU, °C
VRAM_FREE_MIN_GIB: float = 1.0  # Минимум свободного VRAM, ГиБ
RAM_FREE_MIN_GIB: float = 2.0  # Минимум свободной RAM, ГиБ
BANDWIDTH_USAGE_LIMIT: float = 0.80  # Порог использования полосы (80%)

# Интервал повторного NAT-probe (ADR-0015, 5 минут).
STUN_REPROBE_INTERVAL_SEC: int = 300

# Период cleanup-job (ADR-0018)
CLEANUP_JOB_INTERVAL_SEC: int = 60  # раз в 60 секунд

# Порог hard delete: нода offline дольше N дней и strikes_count==0 → удалить.
CLEANUP_HARD_DELETE_DAYS: int = 30

# Hole punching (ADR-0027, §3.2)
# Таймаут подключения при hole punch (3с — время на simultaneous open).
CONNECT_TIMEOUT_SEC: int = 3

# Задержка до rendezvous после получения ответа от трекера (мс).
# Даёт время обоим пирам выставить SYN до начала simultaneous open.
PUNCH_RENDEZVOUS_DELAY_MS: int = 500

# TTL Redis-ключа punch_pending:{peer_id} (сек).
# Нода B забирает pending при следующем heartbeat (раз в HEARTBEAT_INTERVAL_SEC = 15с).
# 10с < 15с → ключ может истечь до heartbeat; увеличено до 20с для надёжности.
PUNCH_NOTIFY_TTL_SEC: int = 20
