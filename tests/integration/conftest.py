"""tests/integration/conftest.py — local fixtures for integration tests.

Heavy-dep mocks (torch, bitsandbytes, aioquic, pynvml, transformers) are
installed as sys.modules stubs only for test_vendor_imports.py, via a
session-scoped autouse fixture.

Scope: autouse is limited to this conftest (tests/integration/), so it does
NOT affect unit tests or app-bootstrap integration tests.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers: build lightweight stub modules
# ---------------------------------------------------------------------------


def _make_torch_stub() -> types.ModuleType:
    """Build a torch stub module with the minimal API used by node/inference.py."""
    torch_mod = types.ModuleType("torch")

    # Tensor
    class Tensor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def to(self, *args: Any, **kwargs: Any) -> Tensor:
            return self

        def __add__(self, other: Any) -> Tensor:
            return self

    torch_mod.Tensor = Tensor  # type: ignore[attr-defined]

    # dtype sentinels
    torch_mod.float16 = "float16"  # type: ignore[attr-defined]
    torch_mod.float32 = "float32"  # type: ignore[attr-defined]
    torch_mod.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    torch_mod.int8 = "int8"  # type: ignore[attr-defined]
    torch_mod.long = "long"  # type: ignore[attr-defined]

    # Scalar functions
    torch_mod.zeros = MagicMock(return_value=Tensor())  # type: ignore[attr-defined]
    torch_mod.ones = MagicMock(return_value=Tensor())  # type: ignore[attr-defined]
    torch_mod.tensor = MagicMock(return_value=Tensor())  # type: ignore[attr-defined]
    torch_mod.load = MagicMock(return_value={})  # type: ignore[attr-defined]
    torch_mod.save = MagicMock()  # type: ignore[attr-defined]
    torch_mod.use_deterministic_algorithms = MagicMock()  # type: ignore[attr-defined]
    torch_mod.device = MagicMock(return_value=object())  # type: ignore[attr-defined]

    # Context managers
    class _NullCtx:
        def __enter__(self) -> _NullCtx:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def __call__(self, fn: Any) -> Any:
            return fn

    torch_mod.no_grad = _NullCtx  # type: ignore[attr-defined]
    torch_mod.inference_mode = _NullCtx  # type: ignore[attr-defined]

    # cuda namespace
    cuda_ns = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        get_device_properties=lambda d: types.SimpleNamespace(
            total_memory=0, major=0, minor=0
        ),
        mem_get_info=lambda d=None: (0, 0),
        empty_cache=MagicMock(),
        synchronize=MagicMock(),
    )
    torch_mod.cuda = cuda_ns  # type: ignore[attr-defined]

    # backends namespace
    torch_mod.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        cuda=types.SimpleNamespace(
            matmul=types.SimpleNamespace(allow_tf32=False),
            allow_tf32=False,
            deterministic=False,
            benchmark=False,
        ),
        cudnn=types.SimpleNamespace(deterministic=False, benchmark=False),
    )
    torch_mod.mps = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]

    # nn sub-module
    nn_mod = types.ModuleType("torch.nn")

    class Module:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def to(self, *args: Any, **kwargs: Any) -> Module:
            return self

        def eval(self) -> Module:
            return self

        def parameters(self) -> list:  # type: ignore[return]
            return []

    class Linear(Module):
        pass

    class Embedding(Module):
        pass

    class LayerNorm(Module):
        pass

    class Parameter(Tensor):
        pass

    class ModuleList(Module):
        def __iter__(self):  # type: ignore[override]
            return iter([])

    class Sequential(Module):
        pass

    nn_mod.Module = Module  # type: ignore[attr-defined]
    nn_mod.Linear = Linear  # type: ignore[attr-defined]
    nn_mod.Embedding = Embedding  # type: ignore[attr-defined]
    nn_mod.LayerNorm = LayerNorm  # type: ignore[attr-defined]
    nn_mod.Parameter = Parameter  # type: ignore[attr-defined]
    nn_mod.ModuleList = ModuleList  # type: ignore[attr-defined]
    nn_mod.Sequential = Sequential  # type: ignore[attr-defined]

    torch_mod.nn = nn_mod  # type: ignore[attr-defined]

    return torch_mod


def _make_transformers_stub() -> dict[str, types.ModuleType]:
    """Build transformers stub modules."""
    mods: dict[str, types.ModuleType] = {}

    # transformers root
    tr_mod = types.ModuleType("transformers")

    class AutoConfig:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> object:
            return object()

    class LlamaForCausalLM:
        pass

    class PretrainedConfig:
        pass

    tr_mod.AutoConfig = AutoConfig  # type: ignore[attr-defined]
    tr_mod.LlamaForCausalLM = LlamaForCausalLM  # type: ignore[attr-defined]
    tr_mod.PretrainedConfig = PretrainedConfig  # type: ignore[attr-defined]
    mods["transformers"] = tr_mod

    # transformers.cache_utils
    cu_mod = types.ModuleType("transformers.cache_utils")

    class DynamicCache:
        pass

    cu_mod.DynamicCache = DynamicCache  # type: ignore[attr-defined]
    mods["transformers.cache_utils"] = cu_mod

    # transformers.models.*
    models_mod = types.ModuleType("transformers.models")
    mods["transformers.models"] = models_mod

    llama_mod = types.ModuleType("transformers.models.llama")
    mods["transformers.models.llama"] = llama_mod

    llama_modeling = types.ModuleType("transformers.models.llama.modeling_llama")

    class LlamaDecoderLayer:
        pass

    class LlamaRMSNorm:
        pass

    class LlamaRotaryEmbedding:
        pass

    llama_modeling.LlamaDecoderLayer = LlamaDecoderLayer  # type: ignore[attr-defined]
    llama_modeling.LlamaRMSNorm = LlamaRMSNorm  # type: ignore[attr-defined]
    llama_modeling.LlamaRotaryEmbedding = LlamaRotaryEmbedding  # type: ignore[attr-defined]
    mods["transformers.models.llama.modeling_llama"] = llama_modeling

    return mods


def _make_bitsandbytes_stub() -> dict[str, types.ModuleType]:
    """Build bitsandbytes stub modules."""
    mods: dict[str, types.ModuleType] = {}

    bnb_mod = types.ModuleType("bitsandbytes")
    nn_bnb = types.ModuleType("bitsandbytes.nn")

    class Linear8bitLt:
        pass

    nn_bnb.Linear8bitLt = Linear8bitLt  # type: ignore[attr-defined]
    bnb_mod.nn = nn_bnb  # type: ignore[attr-defined]

    mods["bitsandbytes"] = bnb_mod
    mods["bitsandbytes.nn"] = nn_bnb

    return mods


def _make_pynvml_stub() -> types.ModuleType:
    """Build pynvml stub module."""
    pynvml_mod = types.ModuleType("pynvml")
    pynvml_mod.nvmlInit = MagicMock()  # type: ignore[attr-defined]
    pynvml_mod.nvmlShutdown = MagicMock()  # type: ignore[attr-defined]
    pynvml_mod.nvmlDeviceGetCount = MagicMock(return_value=0)  # type: ignore[attr-defined]
    pynvml_mod.nvmlDeviceGetHandleByIndex = MagicMock(return_value=object())  # type: ignore[attr-defined]
    pynvml_mod.nvmlDeviceGetMemoryInfo = MagicMock(  # type: ignore[attr-defined]
        return_value=types.SimpleNamespace(total=0, free=0, used=0)
    )
    pynvml_mod.nvmlDeviceGetTemperature = MagicMock(return_value=0)  # type: ignore[attr-defined]
    pynvml_mod.NVMLError = Exception  # type: ignore[attr-defined]
    pynvml_mod.NVML_TEMPERATURE_GPU = 0  # type: ignore[attr-defined]
    return pynvml_mod


def _make_aioquic_stubs() -> dict[str, types.ModuleType]:
    """Build aioquic stub modules."""
    mods: dict[str, types.ModuleType] = {}
    for mod_name in [
        "aioquic",
        "aioquic.asyncio",
        "aioquic.asyncio.client",
        "aioquic.asyncio.server",
        "aioquic.quic",
        "aioquic.quic.configuration",
        "aioquic.quic.connection",
        "aioquic.quic.events",
    ]:
        mods[mod_name] = types.ModuleType(mod_name)

    # QuicConfiguration
    class QuicConfiguration:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    mods["aioquic.quic.configuration"].QuicConfiguration = QuicConfiguration  # type: ignore[attr-defined]

    # Events
    for event_cls in ["StreamDataReceived", "StreamReset", "ConnectionTerminated"]:
        setattr(
            mods["aioquic.quic.events"],
            event_cls,
            type(event_cls, (), {}),
        )

    return mods


# ---------------------------------------------------------------------------
# Autouse fixture: install heavy-dep stubs for test_vendor_imports module
# ---------------------------------------------------------------------------

_HEAVY_DEP_STUBS_INSTALLED: bool = False
_SAVED_MODULES: dict[str, types.ModuleType | None] = {}


@pytest.fixture(scope="session", autouse=True)
def _install_heavy_dep_stubs() -> Generator[None, None, None]:
    """Install sys.modules stubs for heavy deps (torch, transformers, etc.).

    Session-scoped and autouse within tests/integration/ only.
    Does NOT touch tests/unit/ or other conftest scopes.

    Strategy: save existing modules (if any), install stubs, yield, restore.
    """
    global _HEAVY_DEP_STUBS_INSTALLED  # noqa: PLW0603

    torch_stub = _make_torch_stub()
    nn_stub = torch_stub.nn  # type: ignore[attr-defined]
    transformers_stubs = _make_transformers_stub()
    bnb_stubs = _make_bitsandbytes_stub()
    pynvml_stub = _make_pynvml_stub()
    aioquic_stubs = _make_aioquic_stubs()

    all_stubs: dict[str, types.ModuleType] = {
        "torch": torch_stub,
        "torch.nn": nn_stub,
        **transformers_stubs,
        **bnb_stubs,
        "pynvml": pynvml_stub,
        **aioquic_stubs,
    }

    # Save existing (shouldn't be installed in test env, but be safe)
    saved: dict[str, types.ModuleType | None] = {}
    for name in all_stubs:
        saved[name] = sys.modules.get(name)

    # Also save and unload any already-imported node/shared modules
    # so they re-import with stubs active
    node_shared_keys = [k for k in sys.modules if k.startswith(("node.", "shared."))]
    for k in node_shared_keys:
        saved[k] = sys.modules.pop(k)
    # Also save top-level node/shared if imported
    for k in ("node", "shared"):
        if k in sys.modules:
            saved[k] = sys.modules.pop(k)

    sys.modules.update(all_stubs)
    _HEAVY_DEP_STUBS_INSTALLED = True

    yield

    # Restore
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    # Remove our stubs (only if they're still our stubs)
    for name, stub in all_stubs.items():
        if sys.modules.get(name) is stub:
            sys.modules.pop(name, None)

    _HEAVY_DEP_STUBS_INSTALLED = False
