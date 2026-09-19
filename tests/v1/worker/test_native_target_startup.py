# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup cache resets must succeed without reclaiming native text state."""

from types import SimpleNamespace

import pytest

from .test_native_target import (
    binding as native_binding_fixture,
)
from .test_native_target import (
    method,
    output_type,  # noqa: F401 -- shared CPU fixture dependency
    ready,
)

binding = native_binding_fixture


@pytest.mark.parametrize("utility", ["reset_mm_cache", "reset_encoder_cache"])
def test_api_cache_reset_preserves_native_bank_and_checks_lifecycle(binding, utility):
    runner = ready(binding)
    request = runner.admit(0, 1)
    binding.backend.ends[request] = 17
    events = list(binding.events)
    worker = SimpleNamespace(model_runner=runner)
    worker_reset = method("vllm/v1/worker/gpu_worker.py", "Worker", utility, {})
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(
            has_unfinished_requests=lambda: False,
            reset_encoder_cache=lambda: None,
        ),
        mm_receiver_cache=None,
        model_executor=SimpleNamespace(**{utility: lambda: worker_reset(worker)}),
    )
    engine_reset = method("vllm/v1/engine/core.py", "EngineCore", utility, {})
    engine_reset(engine)
    assert binding.backend.ends == {request: 17}
    assert binding.events == events
    runner.shutdown()
    with pytest.raises(RuntimeError, match="not ready"):
        worker_reset(worker)
