# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for admission lifecycle across cancellation and slot reuse.

Purpose: pin request-identity behavior the multi-request scheduler relies on:
a cancelled step is regrantable with the same admission handles, a finished
slot readmits under a reused request_id without inheriting the prior extent,
and a consumed revocation receipt never suppresses the successor's release.
Failures: leaked committed extents, duplicate or missing native frees, stale
generation handles. Fake native completion only; no CUDA or sampler parity.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

BASE_SPEC = importlib.util.spec_from_file_location(
    "native_target_lifecycle_base", Path(__file__).with_name("test_native_target.py")
)
assert BASE_SPEC and BASE_SPEC.loader
base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = base
BASE_SPEC.loader.exec_module(base)

native = base.native
output_type = base.output_type
binding = base.binding


def step_of(step, grants):
    return SimpleNamespace(
        native_target=native.NativeSchedule(17, step, grants),
        num_scheduled_tokens={g.request_id: g.scheduled_rows for g in grants},
        total_num_scheduled_tokens=sum(g.scheduled_rows for g in grants),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )


def test_cancelled_step_regrants_same_admissions_without_readmission(binding):
    runner = base.ready(binding)
    work = base.schedule(runner, lanes=2)
    runner.execute_model(work)
    runner.cancel_step()
    for grant in work.native_target.grants:
        assert binding.backend.ends[grant.request] == 0
    assert not any(
        isinstance(event, tuple) and event[0] == "free" for event in binding.events
    )
    runner.execute_model(step_of(2, work.native_target.grants))
    output = runner.sample_tokens(None)
    assert output.native_target.committed_ends == (("0", 3), ("1", 3))
    assert binding.backend.generations == {0: 1, 1: 1}
    for grant in work.native_target.grants:
        runner.release(grant.request)
    assert binding.events.count(("free", 0)) == binding.events.count(("free", 1)) == 1
    runner.shutdown()


def test_request_id_reuse_after_release_keeps_extents_independent(binding):
    runner = base.ready(binding)
    work = base.schedule(runner, lanes=2)
    runner.execute_model(work)
    assert runner.sample_tokens(None).native_target.committed_ends == (
        ("0", 3),
        ("1", 3),
    )
    finished = work.native_target.grants[0].request
    survivor = work.native_target.grants[1].request
    runner.release(finished)
    assert binding.events.count(("free", 0)) == 1
    fresh = runner.admit(0, 9)
    assert fresh.slot == finished.slot
    assert fresh.generation == finished.generation + 1
    assert fresh != finished
    grants = (
        native.NativeGrant("0", fresh, 0, 0, (0, 1, 2), (2,), "prefill", 0),
        native.NativeGrant("1", survivor, 1, 3, (7,), (0,), "decode", 0),
    )
    runner.execute_model(step_of(2, grants))
    output = runner.sample_tokens(None)
    assert output.native_target.committed_ends == (("0", 3), ("1", 4))
    assert binding.backend.ends[fresh] == 3
    assert binding.backend.ends[survivor] == 4
    assert finished not in binding.backend.ends
    runner.release(fresh)
    runner.release(survivor)
    runner.shutdown()


def test_consumed_revocation_receipt_never_suppresses_successor_free(binding):
    runner = base.stream_ready(binding)
    work = base.stream_work(runner)
    old = work.native_target.grants[0].request
    runner.execute_model(work)
    runner.cancel_step()
    runner.release(old)
    assert ("free", 0) not in binding.events
    reentry = base.schedule(runner, step=2)
    fresh = reentry.native_target.grants[0].request
    assert fresh.slot == old.slot and fresh.generation == old.generation + 1
    runner.execute_model(reentry)
    assert runner.sample_tokens(None).native_target.committed_ends == (("0", 3),)
    runner.release(fresh)
    assert binding.events.count(("free", 0)) == 1
    runner.shutdown()
