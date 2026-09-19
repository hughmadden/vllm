# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped ticket ownership before live C4 tests.

One lane ticket owns multiple requests and one logits lease. The worker must
restore scheduler order, commit each member's accepted prefix, and retain all
members until its shared consumer fence drains. Native computation is stubbed.
"""

from dataclasses import replace

import pytest

from .test_native_target import binding as native_binding_fixture
from .test_native_target import (
    config,
    native,
    output_type,  # noqa: F401 -- fixture dependency
    schedule,
)

binding = native_binding_fixture


def batch_work(binding, speculative=False):
    cfg = config()
    cfg.additional_config["afd_native_target"].update(
        batch_tokens=256, slots=4, decode_batching=True
    )
    if speculative:
        cfg.additional_config["afd_native_target"]["dspark"] = {
            "draft_limit": 3,
            "adaptive": False,
            "confidence_cutoff": None,
        }
    cfg.model_config.max_model_len = 1024
    cfg.scheduler_config.max_num_batched_tokens = 256
    runner = native.NativeTargetRunner(cfg, "cuda:0", binding, owner=17)
    runner.load_model()
    work = schedule(runner, rows=1, lanes=4)
    grants = []
    for grant in work.native_target.grants:
        binding.backend.ends[grant.request] = 10
        grants.append(
            replace(
                grant,
                lane=grant.lane % 2,
                committed_end=10,
                tokens=(7,),
                selected=tuple(range(4)) if speculative else (0,),
                kind="decode",
                phase="dspark" if speculative else "full_target",
                verification_rows=4 if speculative else 0,
            )
        )
    work.native_target = replace(
        work.native_target, grants=tuple(grants), batching=True
    )
    work.num_scheduled_tokens = {g.request_id: g.scheduled_rows for g in grants}
    work.total_num_scheduled_tokens = sum(work.num_scheduled_tokens.values())

    def submit(group):
        binding.events.append(("submit_batch", group.lane, len(group.members)))
        binding.backend.active[group.lane] = group
        return group.lane

    def propose(group):
        ticket = submit(group)
        members, offset = [], 0
        for grant in group.members:
            tokens = (7, 8, 9)
            members.append(
                native.NativeBatchMember(
                    grant.request, tokens, (0, 1, 2), offset, offset
                )
            )
            offset += 3
        return native.NativeBatchProposal(ticket, tuple(members), 11)

    def commit(ticket, accepted):
        assert ticket not in binding.backend.leases
        binding.events.append(("commit_batch", ticket, accepted))
        group = binding.backend.active.pop(ticket)
        for grant, count in zip(group.members, accepted):
            binding.backend.ends[grant.request] += count
        return tuple(binding.backend.ends[g.request] for g in group.members)

    def cancel(ticket):
        assert ticket not in binding.backend.leases
        binding.events.append(("cancel_batch", ticket))
        binding.backend.active.pop(ticket)
        return False

    binding.backend.submit_batch = submit
    binding.backend.propose_batch = propose
    binding.backend.commit_batch = commit
    binding.backend.cancel = cancel
    return runner, work


def test_four_requests_execute_two_owned_batches_and_restore_scheduler_output_order(
    binding,
):
    runner, work = batch_work(binding)
    runner.execute_model(work)
    assert [
        e for e in binding.events if isinstance(e, tuple) and e[0] == "submit_batch"
    ] == [("submit_batch", 0, 2), ("submit_batch", 1, 2)]
    result = runner.sample_tokens(None)
    assert result.req_ids == ["0", "1", "2", "3"]
    assert result.native_target.committed_ends == tuple((str(i), 11) for i in range(4))
    assert not binding.backend.active
    runner.shutdown()


def test_batch_speculation_commits_distinct_accepted_prefixes_by_native_member_order(
    binding,
):
    runner, work = batch_work(binding, speculative=True)
    binding.accepted = (2, 1, 2, 1)
    binding.generated = [[8, 6], [6], [8, 6], [6]]
    runner.execute_model(work)
    result = runner.sample_tokens(None)
    assert result.native_target.committed_ends == (
        ("0", 12),
        ("1", 11),
        ("2", 12),
        ("3", 11),
    )
    assert ("commit_batch", 0, (2, 2)) in binding.events
    assert ("commit_batch", 1, (1, 1)) in binding.events
    runner.shutdown()


def test_shared_batch_fence_blocks_release_of_every_member_until_retry(binding):
    runner, work = batch_work(binding)
    runner.execute_model(work)
    binding.backend.fail_drain = True
    with pytest.raises(RuntimeError, match="consumer event"):
        runner.cancel_step()
    assert len(binding.backend.active) == 2
    assert len(binding.backend.ends) == 4
    with pytest.raises(RuntimeError, match="still active"):
        runner.release(work.native_target.grants[2].request)
    binding.backend.fail_drain = False
    runner.cancel_step()
    assert not binding.backend.active
    assert set(binding.backend.ends.values()) == {10}
    runner.shutdown()


def test_batch_revocation_receipt_prevents_second_release_for_every_member(binding):
    runner, work = batch_work(binding)

    def revoke(ticket):
        group = binding.backend.active.pop(ticket)
        for grant in group.members:
            del binding.backend.ends[grant.request]
        return True

    binding.backend.cancel = revoke
    runner.execute_model(work)
    runner.cancel_step()
    for grant in work.native_target.grants:
        runner.release(grant.request)
    assert not binding.backend.ends
    assert not any(isinstance(e, tuple) and e[0] == "free" for e in binding.events)
    runner.shutdown()


@pytest.mark.parametrize("bad_field", ["offset", "request"])
def test_wrong_batch_proposal_offsets_cancel_ownership_before_execution(
    binding, bad_field
):
    runner, work = batch_work(binding, speculative=True)
    propose = binding.backend.propose_batch

    def invalid(group):
        receipt = propose(group)
        changes = (
            {"output_offset": 0}
            if bad_field == "offset"
            else {"request": receipt.members[0].request}
        )
        return replace(
            receipt,
            members=(receipt.members[0], replace(receipt.members[1], **changes)),
        )

    binding.backend.propose_batch = invalid
    with pytest.raises(RuntimeError, match="batch proposal"):
        runner.execute_model(work)
    assert not binding.backend.active
    assert set(binding.backend.ends.values()) == {10}
    runner.shutdown()


@pytest.mark.parametrize("change", ["prefill", "row_capacity", "step_budget"])
def test_batch_capacity_or_prefill_rejected_before_any_native_submit(binding, change):
    runner, work = batch_work(binding)
    if change == "prefill":
        work.native_target = replace(
            work.native_target,
            grants=(
                replace(work.native_target.grants[0], kind="prefill"),
                *work.native_target.grants[1:],
            ),
        )
    elif change == "row_capacity":
        runner.config.additional_config["afd_native_target"]["batch_tokens"] = 1
    else:
        runner.config.scheduler_config.max_num_batched_tokens = 3
    with pytest.raises(ValueError):
        runner.execute_model(work)
    assert not binding.backend.active
    runner.shutdown()
