# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proposal ownership and accepted-only publication, using a fake native draft.

The scheduler owns an E-row verification grant before tokens exist; native
returns an owned M<=E proposal. vLLM samples C<=M outputs and native commits C
input rows. These CPU tests catch unowned tickets, bad bounds and suffix leaks.
"""

from dataclasses import replace

import pytest

from .test_native_target import (
    binding as native_binding_fixture,
)
from .test_native_target import (
    config,
    native,
    output_type,  # noqa: F401 -- shared CPU fixture dependency
    schedule,
)

binding = native_binding_fixture


def speculative_work(binding, drafts=(8, 9), rows=4):
    cfg = config()
    cfg.additional_config["afd_native_target"].update(
        dspark={"draft_limit": 3, "adaptive": False, "confidence_cutoff": None},
        batch_tokens=256,
    )
    cfg.model_config.max_model_len = 1024
    cfg.scheduler_config.max_num_batched_tokens = 256
    runner = native.NativeTargetRunner(cfg, "cuda:0", binding, owner=17)
    runner.load_model()
    work = schedule(runner, rows=1)
    request = work.native_target.grants[0].request
    binding.backend.ends[request] = 10
    grant = replace(
        work.native_target.grants[0],
        committed_end=10,
        tokens=(7,),
        selected=tuple(range(rows)),
        kind="decode",
        phase="dspark",
        verification_rows=rows,
    )
    work.native_target = replace(work.native_target, grants=(grant,))
    work.num_scheduled_tokens = {"0": rows}
    work.total_num_scheduled_tokens = rows

    def propose(grant):
        tokens = (grant.tokens[0], *drafts)
        ticket = binding.backend.submit(
            replace(grant, tokens=tokens, selected=tuple(range(len(tokens))))
        )
        return native.NativeProposal(ticket, tokens, 17)

    binding.backend.propose = propose
    return runner, work


def test_native_owned_proposal_is_verified_then_only_accepted_prefix_commits(binding):
    runner, work = speculative_work(binding)
    binding.accepted = (2,)
    binding.generated = [[8, 6]]
    runner.execute_model(work)
    assert binding.backend.active[0].tokens == (7, 8, 9)
    result = runner.sample_tokens(None)
    assert result.sampled_token_ids == [[8, 6]]
    assert result.native_target.committed_ends == (("0", 12),)
    assert result.native_target.speculative_counts == (("0", 2, 1),)
    runner.shutdown()


def test_shortened_native_proposal_and_anchor_only_fallback_use_actual_counts(binding):
    runner, work = speculative_work(binding, drafts=())
    binding.generated = [[6]]
    runner.execute_model(work)
    result = runner.sample_tokens(None)
    assert result.native_target.committed_ends == (("0", 11),)
    assert result.native_target.speculative_counts == (("0", 0, 0),)
    runner.shutdown()


def test_oversized_proposal_is_cancelled_before_execute_and_keeps_prior_frontier(
    binding,
):
    runner, work = speculative_work(binding, drafts=(1, 2, 3, 4))
    with pytest.raises(RuntimeError, match="proposal"):
        runner.execute_model(work)
    assert not binding.backend.active
    assert list(binding.backend.ends.values()) == [10]
    assert not any(isinstance(e, tuple) and e[0] == "execute" for e in binding.events)
    runner.shutdown()


@pytest.mark.parametrize("accepted", [0, 2])
def test_zero_acceptance_or_output_count_mismatch_cannot_publish(binding, accepted):
    runner, work = speculative_work(binding)
    binding.accepted = (accepted,)
    binding.generated = [[6]]
    runner.execute_model(work)
    with pytest.raises(RuntimeError, match="sampler"):
        runner.sample_tokens(None)
    assert list(binding.backend.ends.values()) == [10]
    runner.shutdown()


def test_cancel_owned_proposal_discards_suffix_and_preserves_prior_frontier(binding):
    runner, work = speculative_work(binding)
    runner.execute_model(work)
    runner.cancel_step()
    assert not binding.backend.active
    assert list(binding.backend.ends.values()) == [10]
    assert not any(isinstance(e, tuple) and e[0] == "commit" for e in binding.events)
    runner.shutdown()


def test_speculative_envelope_must_fit_configured_limit_and_native_context(binding):
    runner, work = speculative_work(binding)
    grant = work.native_target.grants[0]
    for changed in (
        replace(grant, verification_rows=5),
        replace(grant, committed_end=1023),
        replace(grant, tokens=(7, 8)),
    ):
        invalid = replace(work.native_target, grants=(changed,))
        work.native_target = invalid
        with pytest.raises(ValueError):
            runner.execute_model(work)
    assert not binding.backend.active
    runner.shutdown()
