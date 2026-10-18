# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill-lane livelock regressions (max_parallel_prefills > 1).

A lane-selected async-load candidate whose allocation fails must not
strand the waiting loop with the remaining lane-holders still holding
lanes (intra-step leak).

A never-admissible rank-(-1) waiting cohort must not monopolize the
lanes across steps: lane aging has to hand a running chunk a lane in a
preempt-capable pass so preemption eventually frees blocks.
"""

import pytest

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


def _partial_prefill_output(req_ids: list[str]) -> ModelRunnerOutput:
    """Model-runner output for a step that only advanced prefills."""
    return ModelRunnerOutput(
        req_ids=list(req_ids),
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _empty_output() -> ModelRunnerOutput:
    return _partial_prefill_output([])


def test_async_admission_failure_releases_lane_for_other_candidates():
    """Failed async lane-holders must not block other lane candidates.

    Three async-load candidates (connector CPU-tier hits) can never pass
    the async admission gate while the running chunk reserves its
    remaining prompt. A failed lane-holder parks and releases its lane,
    lanes rotate to the chunk, every lane-holding candidate is processed
    within the step, and the chunk keeps advancing every step.
    """
    scheduler = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=1024,
        max_parallel_prefills=2,
        num_blocks=48,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=False,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=True),
    )
    looked_up: list[str] = []

    def fake_lookup(request, num_computed_tokens):
        looked_up.append(request.request_id)
        if request.request_id == "chunk":
            return 0, False
        # Whole-prompt CPU-tier hit: admission pays the full-sequence
        # requirement and must fail against the chunk's reservation.
        return 240, True

    assert scheduler.connector is not None
    scheduler.connector.get_num_new_matched_tokens = fake_lookup

    (chunk,) = create_requests(
        num_requests=1,
        num_tokens=640,
        req_ids=["chunk"],
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(chunk)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens.get("chunk", 0) == 64
    scheduler.update_from_output(output, _partial_prefill_output(["chunk"]))
    assert chunk.is_prefill_chunk

    for req_id in ("r1", "r2", "r3"):
        (req,) = create_requests(
            num_requests=1,
            num_tokens=640,
            req_ids=[req_id],
            block_size=BLOCK_SIZE,
        )
        scheduler.add_request(req)

    for _ in range(3):
        looked_up.clear()
        output = scheduler.schedule()
        assert output.num_scheduled_tokens.get("chunk", 0) == 64, (
            "running chunk starved: failed async lane-holders kept the lanes"
        )
        # Every lane-holding candidate was processed this step: the waiting
        # loop did not break with unprocessed lane-holders in place.
        assert set(looked_up) == {"r1", "r2", "r3"}
        scheduler.update_from_output(output, _partial_prefill_output(["chunk"]))

    for req_id in ("r1", "r2", "r3"):
        assert scheduler.requests[req_id].status == RequestStatus.WAITING
    assert chunk.num_computed_tokens == 64 + 3 * 64


def test_never_admissible_cohort_cannot_starve_runnable_chunk_forever():
    """A rank-(-1) cohort must age out so preemption can fire.

    A filler plus two running prefill chunks own the whole pool
    (free == 0, their blocks non-evictable) and four promoted-waiting
    requests can never be admitted; lane aging rotates the cohort
    behind the chunks within LANE_AGING_STEPS steps, a chunk reaches
    the preempt-capable first pass, and preemption of the last running
    request frees blocks for a forward.
    """

    scheduler = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=32,
        max_model_len=1024,
        max_parallel_prefills=4,
        num_blocks=33,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=False,
    )
    # Filler: prefills to completion, then holds its blocks pinned
    # (non-evictable) without contributing tokens (no sampled output).
    (filler,) = create_requests(
        num_requests=1,
        num_tokens=128,
        req_ids=["filler"],
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(filler)
    while filler.num_computed_tokens < 128:
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens > 0
        scheduler.update_from_output(
            output, _partial_prefill_output(list(output.num_scheduled_tokens))
        )
    assert filler.num_computed_tokens == 128

    c1, c2 = create_requests(
        num_requests=2,
        num_tokens=320,
        req_ids=["c1", "c2"],
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(c1)
    scheduler.add_request(c2)

    # Fill the pool with the two chunks (lane token-split keeps them even;
    # 24 blocks remain, both need 20 -> neither completes).
    while scheduler.kv_cache_manager.block_pool.get_num_free_blocks() > 0:
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens > 0
        scheduler.update_from_output(
            output, _partial_prefill_output(list(output.num_scheduled_tokens))
        )
    assert c1.is_prefill_chunk and c2.is_prefill_chunk

    # Promoted-waiting cohort: async loads completed (num_computed preset),
    # their blocks were never allocated, and free == 0 makes them
    # permanently inadmissible.
    for i in range(4):
        (promoted,) = create_requests(
            num_requests=1,
            num_tokens=256,
            req_ids=[f"p{i}"],
            block_size=BLOCK_SIZE,
        )
        promoted.num_computed_tokens = 240
        scheduler.add_request(promoted)

    # The intra-step rotation alone cannot recover: while the cohort holds
    # rank -1 at step start, the chunks only reach the preemption-forbidden
    # post-waiting pass, so steps stay empty.
    for _ in range(10):
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens == 0
        scheduler.update_from_output(output, _empty_output())

    # Aging must hand the chunks a lane in the preempt-capable first pass
    # within a bounded number of steps.
    progressed = False
    for _ in range(64):
        output = scheduler.schedule()
        if output.total_num_scheduled_tokens > 0:
            progressed = True
            break
        scheduler.update_from_output(output, _empty_output())
    assert progressed, (
        "never-scheduled rank-(-1) cohort starved the running chunks "
        "forever: lane aging did not rotate lanes to a preempt-capable pass"
    )
    assert c1.num_preemptions + c2.num_preemptions >= 1, (
        "expected preemption of a running chunk to free blocks"
    )
