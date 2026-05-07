"""Stage 3: точка входа ноды.

CLI: python -m node.main --config configs/dev-node.toml

Поток:
  1. Загрузить конфиг (TOML).
  2. Загрузить или сгенерировать Ed25519 identity.
  3. Запустить NAT probe (STUN) — определить публичный endpoint и nat_type (ADR-0015).
  4. Зарегистрироваться на трекере с multiaddr; пройти challenge-response (ADR-0020)
     → session_token + assigned_layers.
  5. Загрузить ModelShard для assigned_layers; fail-fast при OOM (ADR-0008).
  6. Запустить QUIC-сервер (node/network.py) на quic_port из конфига.
  7. Параллельно крутить heartbeat-loop (каждые HEARTBEAT_INTERVAL_SEC секунд).
  8. Параллельно крутить NAT re-probe (каждые STUN_REPROBE_INTERVAL_SEC).
  9. Graceful shutdown на SIGINT/SIGTERM.

Stage 2 pipeline HTTP-сервер (node/pipeline.py) удалён в Stage 4 S4-D3. Используется QUIC-транспорт.

Backoff: exponential, base=HEARTBEAT_INTERVAL_SEC, multiplier=2, max=60 сек.
При 401/404 — перерегистрация, продолжение цикла.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from pathlib import Path

import httpx

from node.config import ConfigError, load_config
from node.inference import ModelShard
from node.monitor import ResourceMonitor
from node.nat import NatProbeLoop, NatProbeResult
from node.network import NodeQuicServer, forward_to_next_hop
from node.tracker_client import (
    TrackerAuthError,
    TrackerClient,
    TrackerNotFoundError,
    canonicalize_tracker_url,
)
from node.trust import IdentityError, load_or_create_identity
from shared.constants import HEARTBEAT_INTERVAL_SEC, STUN_REPROBE_INTERVAL_SEC
from shared.crypto import sign_envelope
from shared.protocol import (
    ForwardEnvelope,
    HeartbeatRequest,
    LayerRange,
    RegisterRequest,
    SessionInit,
    format_multiaddr,
    parse_chain,
    serialize_chain,
)
from shared.quant import dequantize_blockwise_int8, quantize_blockwise_int8


class TrackerProtocolError(RuntimeError):
    """Фатальная несовместимость протокола с трекером.

    Поднимается при обнаружении legacy-поведения трекера (pre-Stage 4),
    когда продолжение работы невозможно. Перехватывается в main() для
    graceful shutdown (QUIC, heartbeat отменяются до выхода).
    """


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Backoff настройки.
_BACKOFF_BASE = HEARTBEAT_INTERVAL_SEC  # 15 сек — минимальный интервал
_BACKOFF_MULTIPLIER = 2
_BACKOFF_MAX = 60  # секунд — потолок backoff при ошибках сети


def _build_multiaddr(cfg, identity, nat_result: NatProbeResult) -> str:
    """Собрать multiaddr ноды для регистрации (ADR-0013, ADR-0015).

    Если STUN дал публичный endpoint — используем его.
    Иначе — fallback на addr из конфига (host:port).
    """
    quic_port = cfg.network.quic_port

    if nat_result.public_endpoint is not None:
        pub_ip, pub_port = nat_result.public_endpoint
        return format_multiaddr("ip4", pub_ip, pub_port, identity.peer_id)

    # Fallback: из конфига, если там уже multiaddr или host:port
    if cfg.addr.startswith("/"):
        # Уже multiaddr
        return cfg.addr
    # host:port → multiaddr
    if ":" in cfg.addr:
        host, _, port_str = cfg.addr.rpartition(":")
        try:
            port = int(port_str)
        except ValueError:
            port = quic_port
    else:
        host = cfg.addr
        port = quic_port
    return format_multiaddr("ip4", host, port, identity.peer_id)


def _build_register_request(cfg, identity, nat_result: NatProbeResult) -> RegisterRequest:
    """Собрать RegisterRequest из конфига, identity и результата NAT probe."""
    addr = _build_multiaddr(cfg, identity, nat_result)
    return RegisterRequest(
        peer_id=identity.peer_id,
        public_key=identity.public_key_b58,
        layers=LayerRange(start=cfg.layers_start, end=cfg.layers_end),
        model_id=cfg.model_id,
        vram_gb=cfg.vram_gb,
        bandwidth_mbps=cfg.bandwidth_mbps,
        addr=addr,
    )


async def _register_and_auth(
    client: TrackerClient, cfg, identity, nat_result: NatProbeResult
) -> str:
    """Зарегистрировать ноду и пройти challenge-response; вернуть session_token.

    Stage 4 (ADR-0020): register → challenge → authenticate → session_token.
    Повторяет попытки бесконечно с exponential backoff при сетевых ошибках.
    При 401 на /auth — fatal (wrong config / key mismatch), sys.exit(1).
    """
    canonical_url = canonicalize_tracker_url(cfg.tracker_url)
    backoff = _BACKOFF_BASE
    while True:
        try:
            req = _build_register_request(cfg, identity, nat_result)
            reg_resp = await client.register(req)
            logger.info(
                "registered as %s (nat=%s, addr=%s)",
                identity.peer_id,
                nat_result.nat_type,
                req.addr,
            )

            # Challenge-response (ADR-0020) — обязателен; трекер без challenge fail-fast.
            if reg_resp.challenge_id is None or reg_resp.challenge is None:
                raise TrackerProtocolError(
                    "Tracker did not return challenge — legacy (pre-Stage 4) trackers are "
                    "no longer supported. Upgrade the tracker to Stage 4+."
                )

            logger.debug("authenticating with challenge_id=%s", reg_resp.challenge_id)
            session_token = await client.authenticate(
                challenge_id=reg_resp.challenge_id,
                challenge_b64=reg_resp.challenge,
                signing_key=identity.signing_key,
                peer_id_raw=identity.peer_id_bytes(),
                tracker_url=canonical_url,
            )
            logger.info(
                "authenticated, session_token obtained; heartbeat every %ds",
                HEARTBEAT_INTERVAL_SEC,
            )
            return session_token

        except TrackerAuthError as exc:
            # 401 при /auth — это фатальная ошибка конфигурации (неверный ключ или URL).
            raise TrackerProtocolError(
                f"Authentication rejected by tracker (401): {exc}. "
                "Check that tracker_url matches LLM_SWARM_TRACKER_PUBLIC_URL on the tracker."
            ) from exc
        except (httpx.HTTPError, httpx.HTTPStatusError) as exc:
            logger.warning("register/auth failed (%s), retry in %ds", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * _BACKOFF_MULTIPLIER, _BACKOFF_MAX)


async def _heartbeat_loop(
    client: TrackerClient,
    cfg,
    identity,
    token: str,
    nat_probe_loop: NatProbeLoop,
    resource_monitor: ResourceMonitor | None = None,
) -> None:
    """Бесконечный heartbeat-loop.

    - Каждые HEARTBEAT_INTERVAL_SEC секунд шлёт HeartbeatRequest.
    - При сетевых ошибках / 5xx — exponential backoff, не падает.
    - При 401 или 404 — перерегистрация (новый register) и продолжение.
    - asyncio.CancelledError — выходим чисто (graceful shutdown).
    - Stage 6 (ADR-0038): включает метрики ResourceMonitor в HeartbeatRequest.
    """
    backoff = _BACKOFF_BASE
    current_token = token

    while True:
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
        except asyncio.CancelledError:
            logger.info("heartbeat loop cancelled, shutting down")
            return

        try:
            # Собираем метрики из ResourceMonitor (Stage 6, ADR-0038)
            degraded = False
            degraded_reasons: list[str] = []
            kv_active_sessions = 0
            kv_used_bytes = 0
            vram_used_bytes: int | None = None
            ram_used_bytes: int | None = None
            gpu_temp: float | None = None
            vram_free_gb: float | None = None

            if resource_monitor is not None:
                snap = resource_monitor.snapshot
                degraded = snap.degraded
                degraded_reasons = list(snap.degraded_reasons)
                kv_active_sessions = snap.kv_active_sessions
                kv_used_bytes = snap.kv_used_bytes
                vram_used_bytes = snap.vram_used_bytes
                ram_used_bytes = snap.ram_used_bytes
                gpu_temp = snap.gpu_temp_c
                if snap.vram_free_bytes is not None:
                    vram_free_gb = snap.vram_free_bytes / (1024**3)

            hb_req = HeartbeatRequest(
                peer_id=identity.peer_id,
                token=current_token,
                served_tokens=0,  # TODO Stage 4: real accounting (§5.2)
                gpu_temp=gpu_temp,
                vram_free_gb=vram_free_gb,
                degraded=degraded,
                degraded_reasons=degraded_reasons,
                kv_active_sessions=kv_active_sessions,
                kv_used_bytes=kv_used_bytes,
                vram_used_bytes=vram_used_bytes,
                ram_used_bytes=ram_used_bytes,
            )
            await client.heartbeat(hb_req)
            logger.debug("heartbeat ok, peer_id=%s degraded=%s", identity.peer_id, degraded)
            # Успешный heartbeat — сбрасываем backoff.
            backoff = _BACKOFF_BASE

        except (TrackerAuthError, TrackerNotFoundError) as exc:
            # Токен протух или нода забыта — полный цикл re-register + re-auth.
            logger.warning("%s, re-registering...", exc)
            try:
                nat_result = nat_probe_loop.latest
                current_token = await _register_and_auth(client, cfg, identity, nat_result)
            except asyncio.CancelledError:
                logger.info("re-registration cancelled, shutting down")
                return

        except asyncio.CancelledError:
            logger.info("heartbeat loop cancelled during request, shutting down")
            return

        except httpx.HTTPError as exc:
            logger.warning(
                "heartbeat error (%s: %s), retry in %ds",
                type(exc).__name__,
                exc,
                backoff,
            )
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                logger.info("heartbeat loop cancelled during backoff, shutting down")
                return
            backoff = min(backoff * _BACKOFF_MULTIPLIER, _BACKOFF_MAX)


def _make_envelope_handler(
    shard: ModelShard,
    signing_key,
    cfg,
    nat_probe_loop: NatProbeLoop,
    session_token: str,
    self_peer_id: str,
    resource_monitor: ResourceMonitor | None = None,
) -> object:
    """Создать envelope_handler для NodeQuicServer.

    Handler вызывается для каждого входящего ForwardEnvelope.
    Выполняет forward pass и либо форвардит следующему хопу, либо возвращает chain.

    Возвращает callable (async function).
    """

    async def handle_envelope(
        env: ForwardEnvelope,
        sig: bytes,
        session_init: SessionInit,
    ) -> bytes | None:
        """Обработать входящий ForwardEnvelope.

        1. Деквантуем активации.
        2. Forward pass через shard.
        3. Строим свой envelope с подписью.
        4. Если есть следующий хоп — форвардим; иначе возвращаем chain.
        """
        import base58 as _base58
        from nacl.signing import VerifyKey

        # Верифицируем подпись предыдущей ноды
        if env.hop_index == 0:
            # Первая нода — нет предыдущей подписи
            pass
        else:
            # Находим pubkey предыдущего хопа из маршрута
            prev_hop_idx = env.hop_index - 1
            if prev_hop_idx < len(session_init.route):
                prev_route = session_init.route[prev_hop_idx]
                try:
                    prev_pk_bytes = _base58.b58decode(prev_route.pubkey_b58)
                    prev_pk = VerifyKey(prev_pk_bytes)
                    from shared.crypto import verify_envelope

                    if not verify_envelope(prev_pk, env, sig):
                        logger.warning(
                            "Invalid envelope signature from hop %d session %s",
                            env.hop_index,
                            session_init.session_id,
                        )
                        return None
                except Exception as exc:
                    logger.warning("Pubkey verification error: %s", exc)

        # Деквантуем активации (или токены для первой ноды)
        if env.hop_index == 0:
            # Первая нода: activations_blob содержит квантованные hidden states
            # после embedding (или raw token_ids как квант? — нет, embedding уже в ноде)
            hidden_states = dequantize_blockwise_int8(env.activations_blob)
        else:
            hidden_states = dequantize_blockwise_int8(env.activations_blob)

        # Stage 6 (ADR-0038): throttle hook на уровне inference handler.
        # Если нода в degraded — отказываем новым сессиям (до begin_session).
        # Проверяем только если сессия ещё не открыта (первый вызов).
        session_is_new = session_init.session_id not in shard._kv_sessions
        if session_is_new and resource_monitor is not None:
            if not resource_monitor.allow_new_session():
                snap = resource_monitor.snapshot
                logger.warning(
                    "ResourceMonitor degraded (inference handler), rejecting session %s: %s",
                    session_init.session_id,
                    snap.degraded_reasons,
                )
                return None

        # Forward pass
        shard.begin_session(session_init.session_id)
        if shard._weight_manager is not None:
            # Lazy loading path: async forward с await _ensure_layer (E3.5)
            result = await shard.forward_async(
                hidden_states=hidden_states,
                session_id=session_init.session_id,
            )
        else:
            # Eager path (random-init / уже загружены): sync в executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: shard.forward(
                    hidden_states=hidden_states,
                    session_id=session_init.session_id,
                ),
            )

        # Квантуем выход
        if "logits" in result:
            out_blob = quantize_blockwise_int8(result["logits"])
        else:
            out_blob = quantize_blockwise_int8(result["hidden_states"])

        # Строим свой envelope
        my_env = ForwardEnvelope(
            session_id=session_init.session_id,
            hop_index=env.hop_index + 1,
            prev_sig=sig,
            layer_start=shard.layer_start,
            layer_end=shard.layer_end,
            activations_blob=out_blob,
        )
        my_sig = sign_envelope(signing_key, my_env)

        # Определяем следующий хоп
        my_hop_idx = session_init.hop_index
        next_hop_idx = my_hop_idx + 1

        if "logits" in result or next_hop_idx >= len(session_init.route):
            # Последняя нода — возвращаем chain из одного envelope
            chain_bytes = serialize_chain([(my_env, my_sig)])
            return chain_bytes

        next_hop = session_init.route[next_hop_idx]
        nat_type = nat_probe_loop.latest.nat_type

        # Обновляем SessionInit для следующего хопа
        next_session_init = SessionInit(
            session_id=session_init.session_id,
            session_token=session_token,
            route=session_init.route,
            hop_index=next_hop_idx,
            peer_id_self=next_hop.peer_id,
        )

        # Форвардим следующему хопу
        try:
            downstream_bytes = await forward_to_next_hop(
                next_hop=next_hop,
                session_init=next_session_init,
                signing_key=signing_key,
                envelope=my_env,
                sig=my_sig,
                tracker_url=cfg.tracker_url,
                nat_type=nat_type,
                self_peer_id=self_peer_id,
                self_session_token=session_token,
            )
        except Exception as exc:
            logger.error(
                "Failed to forward to next hop %s: %s",
                next_hop.peer_id,
                exc,
            )
            return None

        # Добавляем свой envelope в начало цепочки
        try:
            downstream_chain = parse_chain(downstream_bytes)
        except ValueError as exc:
            logger.error("Invalid chain from next hop %s: %s", next_hop.peer_id, exc)
            return None

        full_chain = [(my_env, my_sig), *downstream_chain]
        full_chain.sort(key=lambda x: x[0].hop_index)
        return serialize_chain(full_chain)

    return handle_envelope


async def main() -> int:
    """Точка входа: CLI → config → identity → NAT probe → register → shard → QUIC → heartbeat.

    Returns:
        0 при штатном завершении, 1 при фатальной ошибке.
    """
    parser = argparse.ArgumentParser(
        description="llm-swarm node — participates in distributed LLM inference"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to TOML node config (e.g. configs/dev-node.toml)",
    )
    parser.add_argument(
        "--weights-dtype",
        choices=["fp16", "int8"],
        default=None,
        metavar="DTYPE",
        help=(
            "Override weights dtype from config: 'fp16' | 'int8'. "
            "Default: value from [weights] weights_dtype in config (int8). "
            "int8 uses bitsandbytes Linear8bitLt for Llama-2/3 (ADR-0034). "
            "Not supported on macOS without CUDA."
        ),
    )
    args = parser.parse_args()

    # --- Конфиг ---
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    # CLI override для weights_dtype (E3.2, ADR-0034)
    weights_dtype = args.weights_dtype or cfg.weights.weights_dtype

    # --- Identity ---
    try:
        identity = load_or_create_identity(cfg.data_dir)
    except IdentityError as exc:
        logger.error("Failed to load identity: %s", exc)
        return 1

    # --- Graceful shutdown setup ---
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _shutdown_handler() -> None:
        logger.info("shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown_handler)

    # --- NAT probe (ADR-0015) ---
    logger.info("Starting NAT probe via STUN servers: %s", cfg.network.stun_servers)
    nat_loop = NatProbeLoop(
        servers=cfg.network.stun_servers,
        interval_sec=STUN_REPROBE_INTERVAL_SEC,
    )
    nat_result = await nat_loop.start()
    logger.info(
        "NAT probe result: nat_type=%s, public_endpoint=%s",
        nat_result.nat_type,
        nat_result.public_endpoint,
    )

    try:
        async with TrackerClient(cfg.tracker_url) as client:
            # --- Первичная регистрация + challenge-response → получаем session_token ---
            token = await _register_and_auth(client, cfg, identity, nat_result)

            # Используем конфиговые значения (Stage 1 echo, ADR-0006).
            layer_start = cfg.layers_start
            layer_end = cfg.layers_end

            # --- WeightManager (E3.1, Stage 5) ---
            from node.weights import WeightManager

            weight_manager = WeightManager(
                cache_dir=cfg.weights.cache_dir,
                tracker_client=client,
                identity=identity,
                fetch_concurrency=cfg.weights.fetch_concurrency,
                upload_concurrency=cfg.weights.upload_concurrency,
                manifest_ttl_seconds=cfg.weights.manifest_ttl_seconds,
                session_token=token,
            )

            # --- Загрузка ModelShard (fail-fast OOM) ---
            logger.info(
                "loading shard: model=%s layers=[%d,%d) weights_dtype=%s",
                cfg.model_id,
                layer_start,
                layer_end,
                weights_dtype,
            )
            shard = ModelShard(
                model_id=cfg.model_id,
                layer_start=layer_start,
                layer_end=layer_end,
                device="cuda" if _cuda_available() else "cpu",
                vram_gb=cfg.vram_gb,
                weight_manager=weight_manager,
                weights_dtype=weights_dtype,
            )
            try:
                shard.load()
            except SystemExit:
                # OOM fail-fast уже залогирован в ModelShard._check_cuda_oom
                raise
            except ValueError as exc:
                logger.error("Failed to load shard: %s", exc)
                return 1

            logger.info("shard loaded, starting QUIC server on port %d", cfg.network.quic_port)

            # --- ResourceMonitor (Stage 6, ADR-0038) ---
            resource_monitor = ResourceMonitor(
                bandwidth_mbps=cfg.bandwidth_mbps,
                shard=shard,
            )
            await resource_monitor.start()

            # --- KV-cache sweep task (Stage 6, ADR-0040) ---
            await shard.start_kv_sweep()

            # --- Envelope handler ---
            envelope_handler = _make_envelope_handler(
                shard=shard,
                signing_key=identity.signing_key,
                cfg=cfg,
                nat_probe_loop=nat_loop,
                session_token=token,
                self_peer_id=identity.peer_id,
                resource_monitor=resource_monitor,
            )

            # --- QUIC server ---
            quic_server = NodeQuicServer(
                host="0.0.0.0",
                port=cfg.network.quic_port,
                signing_key=identity.signing_key,
                envelope_handler=envelope_handler,
                resource_monitor=resource_monitor,
            )
            await quic_server.start()

            # --- Запуск heartbeat и NAT re-probe параллельно ---
            heartbeat_task = asyncio.create_task(
                _heartbeat_loop(
                    client,
                    cfg,
                    identity,
                    token,
                    nat_loop,
                    resource_monitor=resource_monitor,
                ),
                name="heartbeat",
            )

            # Ждём сигнала завершения.
            await shutdown_event.wait()

            # Отменяем tasks.
            for task in (heartbeat_task,):
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            await nat_loop.stop()
            await quic_server.stop()
            await resource_monitor.stop()
            await shard.stop_kv_sweep()

    except TrackerProtocolError as exc:
        logger.error("Fatal tracker protocol error: %s", exc)
        return 1

    logger.info("node stopped cleanly")
    return 0


def _cuda_available() -> bool:
    """Проверить доступность CUDA."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    asyncio.run(main())
