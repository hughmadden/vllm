"""CPU-only config/factory selection; constructors are explicit test injections.

The real production binding/getter run. This does not initialize a model or CUDA.
"""
from types import SimpleNamespace as NS

import pytest

from vllm.v1.worker import native_target as native
from vllm.v1.worker.native_target_binding import RetainedNativeBinding
from vllm_afd.native_target import client, composite

MISSING = object()


def make_config(tmp_path, flag=MISSING):
    options = dict(
        enabled=True, implementation="retained", peers=["127.0.0.1:1"] * 4,
        batch_tokens=1024, slots=2, source_pool_budget_bytes=64 * 1024 * 1024,
        prefill_mode="encoder_stream", decode_batching=True,
    )
    for key in ("abi_library", "snapshot", "native_lib"):
        path = tmp_path / key
        path.touch()
        options[key] = str(path)
    if flag is not MISSING:
        options["composite_drain_commit"] = flag
    return NS(
        additional_config={"afd_native_target": options}, use_v2_model_runner=True,
        parallel_config=NS(world_size=1, data_parallel_size=1,
            distributed_executor_backend="uni", enable_dbo=False,
            enable_fault_tolerance=False),
        scheduler_config=NS(max_num_batched_tokens=1024, max_num_seqs=2,
            async_scheduling=False),
        model_config=NS(max_model_len=4096, get_vocab_size=lambda: 129280,
            hf_config=NS(hidden_size=5120, num_hidden_layers=40),
            enable_return_routed_experts=False, return_sampling_mask=False,
            runner_type="generate", enable_sleep_mode=False),
        cache_config=NS(kv_cache_memory_bytes=None, enable_prefix_caching=False),
        speculative_config=None, lora_config=None, kv_transfer_config=None,
        ec_transfer_config=None, weight_transfer_config=None,
    )


@pytest.mark.parametrize("flag", [None, 0, 1, "true", "false", [], {}, 1.0])
def test_invalid_composite_flag_fails_before_client_creation(tmp_path, monkeypatch, flag):
    config = make_config(tmp_path, flag)
    monkeypatch.setattr(native, "_binding", None)
    binding = RetainedNativeBinding()
    for action in (
        lambda: binding.validate_config(config),
        lambda: native.get_native_target_binding(config),
        lambda: binding.create_backend(config, NS(type="cuda", index=0), 7001),
    ):
        with pytest.raises(ValueError, match="composite_drain_commit must be boolean"):
            action()


@pytest.mark.parametrize("flag,expected", [(MISSING, False), (False, False), (True, True)])
def test_factory_selects_original_or_packaged_client(tmp_path, monkeypatch, flag, expected):
    config = make_config(tmp_path, flag)
    monkeypatch.setattr(native, "_binding", None)
    binding = native.get_native_target_binding(config)
    assert type(binding) is RetainedNativeBinding
    calls = []

    class BaseConstructor(client.NativeTargetClient):
        def __init__(self, target, **kwargs):
            calls.append((type(self), target, kwargs))
            self.negotiations = 0
        def initialize(self):
            pass

    class CompositeConstructor(BaseConstructor):
        def negotiate_composite(self):
            self.negotiations += 1

    # Explicit constructor injection; production create_backend is NOT replaced.
    monkeypatch.setattr(client, "NativeTargetClient", BaseConstructor)
    monkeypatch.setattr(composite, "CompositeNativeTargetClient", CompositeConstructor)
    backend = binding.create_backend(config, NS(type="cuda", index=0), 7001)
    backend.initialize()
    selected, target, kwargs = calls[0]
    assert selected is (CompositeConstructor if expected else BaseConstructor)
    assert backend._composite_enabled is expected
    assert backend.client.negotiations == int(expected)
    assert target["owner"] == 7001 and target["max_context_tokens"] == 4096
    assert "composite_drain_commit" not in target
    assert kwargs["enabled"] is True and kwargs["device_id"] == 0
    assert kwargs["library_path"] == config.additional_config["afd_native_target"]["abi_library"]


def test_reused_registered_binding_does_not_latch_previous_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "_binding", None)
    on = make_config(tmp_path, True)
    binding = native.get_native_target_binding(on)
    off = make_config(tmp_path, False)
    default = make_config(tmp_path)
    assert native.get_native_target_binding(off) is binding
    assert native.get_native_target_binding(default) is binding
    assert binding._composite_requested(on.additional_config["afd_native_target"]) is True
    assert binding._composite_requested(off.additional_config["afd_native_target"]) is False
    assert binding._composite_requested(default.additional_config["afd_native_target"]) is False


def test_disabled_target_returns_no_binding(tmp_path, monkeypatch):
    config = make_config(tmp_path, True)
    config.additional_config["afd_native_target"]["enabled"] = False
    monkeypatch.setattr(native, "_binding", None)
    assert native.get_native_target_binding(config) is None


def test_composite_does_not_enable_speculative_configuration(tmp_path, monkeypatch):
    config = make_config(tmp_path, True)
    config.speculative_config = NS(model="not-loaded")
    monkeypatch.setattr(native, "_binding", None)
    with pytest.raises(native.NativeTargetUnavailable, match="ordinary draft models"):
        native.get_native_target_binding(config)
