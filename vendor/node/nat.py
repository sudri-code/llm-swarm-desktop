"""Stage 3: NAT probe через публичные STUN-серверы (ADR-0015, §4.4).

Классификация NAT:
  - public   — публичный endpoint совпадает с локальным.
  - cone     — full/restricted/port-restricted cone; hole punching работает.
  - symmetric — публичный port меняется per-destination → relay-only.
  - unknown  — STUN недоступен или классификация не сошлась → relay-only.

Fail-soft: при недоступности всех STUN нода продолжает работу в relay-only
режиме (unknown). Реальный сетевой STUN не дёргается в unit-тестах — функция
принимает mock-callable через параметр _stun_query.

Re-probe раз в STUN_REPROBE_INTERVAL_SEC.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import struct
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Literal

from shared.constants import (
    STUN_PROBE_TIMEOUT_SEC,
    STUN_REPROBE_INTERVAL_SEC,
    STUN_SERVERS_DEFAULT,
)

logger = logging.getLogger(__name__)

NatType = Literal["public", "cone", "symmetric", "unknown"]

# STUN magic cookie (RFC 5389)
_STUN_MAGIC = 0x2112A442
# STUN Binding request type
_STUN_BINDING_REQUEST = 0x0001
_STUN_BINDING_RESPONSE = 0x0101
# XOR-MAPPED-ADDRESS attribute type
_ATTR_XOR_MAPPED_ADDRESS = 0x0020
# MAPPED-ADDRESS attribute (RFC 3489 fallback)
_ATTR_MAPPED_ADDRESS = 0x0001


@dataclass(frozen=True)
class NatProbeResult:
    """Результат NAT probe (ADR-0015).

    Атрибуты:
        public_endpoint: (ip, port) публичного endpoint'а; None если STUN недоступен.
        nat_type: классификация NAT.
    """

    public_endpoint: tuple[str, int] | None
    nat_type: NatType


def _build_stun_request(transaction_id: bytes) -> bytes:
    """Сформировать STUN Binding Request (RFC 5389)."""
    assert len(transaction_id) == 12, "transaction_id must be 12 bytes"
    # Header: type(2) + length(2) + magic(4) + txn_id(12)
    return struct.pack(">HHI", _STUN_BINDING_REQUEST, 0, _STUN_MAGIC) + transaction_id


def _parse_stun_response(data: bytes, transaction_id: bytes) -> tuple[str, int] | None:
    """Разобрать STUN Binding Response и вернуть (ip, port) или None при ошибке."""
    if len(data) < 20:
        return None

    msg_type, _msg_len, magic = struct.unpack_from(">HHI", data, 0)
    if msg_type != _STUN_BINDING_RESPONSE:
        return None
    if magic != _STUN_MAGIC:
        return None
    txn = data[8:20]
    if txn != transaction_id:
        return None

    # Парсим атрибуты
    offset = 20
    xor_mapped: tuple[str, int] | None = None
    mapped: tuple[str, int] | None = None

    while offset + 4 <= len(data):
        attr_type, attr_len = struct.unpack_from(">HH", data, offset)
        offset += 4
        attr_val = data[offset : offset + attr_len]
        # Выравнивание до 4 байт
        offset += (attr_len + 3) & ~3

        if attr_type == _ATTR_XOR_MAPPED_ADDRESS and len(attr_val) >= 8:
            family = attr_val[1]
            if family == 0x01:  # IPv4
                port = struct.unpack_from(">H", attr_val, 2)[0] ^ (_STUN_MAGIC >> 16)
                raw_ip = struct.unpack_from(">I", attr_val, 4)[0] ^ _STUN_MAGIC
                ip = str(ipaddress.IPv4Address(raw_ip))
                xor_mapped = (ip, port)
        elif attr_type == _ATTR_MAPPED_ADDRESS and len(attr_val) >= 8:
            family = attr_val[1]
            if family == 0x01:
                port = struct.unpack_from(">H", attr_val, 2)[0]
                raw_ip = struct.unpack_from(">I", attr_val, 4)[0]
                ip = str(ipaddress.IPv4Address(raw_ip))
                mapped = (ip, port)

    return xor_mapped or mapped


class _UdpStunProtocol(asyncio.DatagramProtocol):
    """asyncio UDP протокол для одного STUN-запроса."""

    def __init__(self, transaction_id: bytes) -> None:
        self._txn_id = transaction_id
        loop = asyncio.get_event_loop()
        self._result: asyncio.Future[tuple[str, int] | None] = loop.create_future()

    def datagram_received(self, data: bytes, addr: object) -> None:
        if not self._result.done():
            parsed = _parse_stun_response(data, self._txn_id)
            self._result.set_result(parsed)

    def error_received(self, exc: Exception) -> None:
        if not self._result.done():
            self._result.set_exception(exc)

    def connection_lost(self, exc: Exception | None) -> None:
        if not self._result.done():
            self._result.set_result(None)

    async def wait(self, timeout: float) -> tuple[str, int] | None:
        try:
            return await asyncio.wait_for(asyncio.shield(self._result), timeout=timeout)
        except TimeoutError:
            return None


async def _stun_query_real(
    host: str,
    port: int,
    timeout: float = STUN_PROBE_TIMEOUT_SEC,
) -> tuple[str, int] | None:
    """Отправить один STUN Binding Request и вернуть (ip, port) или None.

    Реальный сетевой запрос — не использовать в unit-тестах.
    """
    import os

    txn_id = os.urandom(12)
    request = _build_stun_request(txn_id)

    loop = asyncio.get_event_loop()
    try:
        # Резолвим hostname
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_DGRAM)
        if not infos:
            return None
        target_addr = infos[0][4]  # (host, port)

        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _UdpStunProtocol(txn_id),
            remote_addr=target_addr,
        )
        try:
            transport.sendto(request)
            return await protocol.wait(timeout)
        finally:
            transport.close()
    except OSError as exc:
        logger.debug("STUN query to %s:%d failed: %s", host, port, exc)
        return None


def _parse_stun_server(server: str) -> tuple[str, int]:
    """Разобрать строку 'host:port' в (host, port)."""
    if ":" not in server:
        raise ValueError(f"STUN server must be 'host:port', got: {server!r}")
    host, _, port_str = server.rpartition(":")
    return host, int(port_str)


def _classify_nat(
    local_ip: str,
    local_port: int,
    endpoint1: tuple[str, int],
    endpoint2: tuple[str, int] | None,
) -> NatType:
    """Классифицировать NAT по двум наблюдениям от разных серверов.

    Алгоритм (упрощённый RFC 5780):
    - Если публичный IP == локальный IP → public.
    - Если endpoint от двух серверов совпадает → cone.
    - Если endpoint разный → symmetric.
    - Если только одно наблюдение → cone (предполагаем лучшее).
    """
    pub_ip, pub_port = endpoint1
    try:
        pub_addr = ipaddress.ip_address(pub_ip)
        loc_addr = ipaddress.ip_address(local_ip)
        if pub_addr == loc_addr and pub_port == local_port:
            return "public"
    except ValueError:
        pass

    if endpoint2 is None:
        return "cone"

    if endpoint1 == endpoint2:
        return "cone"

    # Разные порты от разных серверов — symmetric
    return "symmetric"


async def probe_nat(
    servers: list[str] | None = None,
    timeout: float = STUN_PROBE_TIMEOUT_SEC,
    _stun_query: Callable[..., Coroutine] | None = None,
) -> NatProbeResult:
    """Определить NAT-тип и публичный endpoint через STUN (ADR-0015).

    Делает round-robin по серверам из `servers` (или STUN_SERVERS_DEFAULT).
    При недоступности всех → nat_type="unknown", public_endpoint=None (fail-soft).

    Args:
        servers: Список STUN-серверов в формате "host:port".
                 По умолчанию STUN_SERVERS_DEFAULT.
        timeout: Таймаут на один запрос в секундах.
        _stun_query: Инжекция для unit-тестов. Если None — реальный сетевой запрос.

    Returns:
        NatProbeResult с public_endpoint и nat_type.
    """
    if servers is None:
        servers = list(STUN_SERVERS_DEFAULT)

    query_fn = _stun_query if _stun_query is not None else _stun_query_real

    # Определяем локальный IP через временный сокет
    local_ip = "0.0.0.0"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        local_port = s.getsockname()[1]
        s.close()
    except OSError:
        local_port = 0

    endpoints: list[tuple[str, int]] = []

    for server in servers:
        try:
            host, port = _parse_stun_server(server)
        except ValueError as exc:
            logger.warning("Invalid STUN server %r: %s", server, exc)
            continue

        try:
            if _stun_query is not None:
                # mock callable: может принимать (host, port) или просто быть callable
                result = await query_fn(host, port, timeout)
            else:
                result = await _stun_query_real(host, port, timeout)
        except Exception as exc:
            logger.debug("STUN query to %s failed: %s", server, exc)
            result = None

        if result is not None:
            logger.debug("STUN %s → %s:%d", server, result[0], result[1])
            endpoints.append(result)
        else:
            logger.debug("STUN %s → no response", server)

    if not endpoints:
        logger.warning(
            "All STUN servers unreachable: %s. Falling back to relay-only (nat_type=unknown).",
            servers,
        )
        return NatProbeResult(public_endpoint=None, nat_type="unknown")

    endpoint1 = endpoints[0]
    endpoint2 = endpoints[1] if len(endpoints) > 1 else None
    nat_type = _classify_nat(local_ip, local_port, endpoint1, endpoint2)

    logger.info(
        "NAT probe complete: public_endpoint=%s:%d, nat_type=%s",
        endpoint1[0],
        endpoint1[1],
        nat_type,
    )
    return NatProbeResult(public_endpoint=endpoint1, nat_type=nat_type)


class NatProbeLoop:
    """Фоновый task: re-probe NAT каждые STUN_REPROBE_INTERVAL_SEC.

    Использование:
        loop = NatProbeLoop(servers=[...])
        await loop.start()
        result = loop.latest          # последний NatProbeResult
        await loop.stop()
    """

    def __init__(
        self,
        servers: list[str] | None = None,
        interval_sec: int = STUN_REPROBE_INTERVAL_SEC,
        timeout: float = STUN_PROBE_TIMEOUT_SEC,
        _stun_query: Callable | None = None,
    ) -> None:
        self._servers = servers
        self._interval = interval_sec
        self._timeout = timeout
        self._stun_query = _stun_query
        self._latest: NatProbeResult = NatProbeResult(public_endpoint=None, nat_type="unknown")
        self._task: asyncio.Task | None = None

    @property
    def latest(self) -> NatProbeResult:
        return self._latest

    async def start(self) -> NatProbeResult:
        """Запустить начальный probe и запустить фоновый цикл."""
        self._latest = await probe_nat(
            servers=self._servers,
            timeout=self._timeout,
            _stun_query=self._stun_query,
        )
        self._task = asyncio.create_task(self._loop(), name="nat-probe-loop")
        return self._latest

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                return
            try:
                self._latest = await probe_nat(
                    servers=self._servers,
                    timeout=self._timeout,
                    _stun_query=self._stun_query,
                )
            except Exception as exc:
                logger.warning("NAT re-probe failed: %s", exc)

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
