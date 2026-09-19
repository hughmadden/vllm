# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU serving tests using real vLLM request/output/sampling types.

Purpose: serve text through native grants without allocating ordinary KV.
I/O: Request -> SchedulerOutput -> ModelRunnerOutput -> EngineCoreOutputs.
Failures: starvation, false publication, canceled output, cache leaks and changed
sampling behavior. The bank/target are fake; the sampler is vLLM's real sampler
on CPU with compilation disabled for its small tensor helpers.
"""

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.native_scheduler import NativeScheduler
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.native_target import (
    NativeCacheInfo,
    NativeGrant,
    NativeProposal,
    NativeRequest,
    NativeSchedule,
    NativeStepResult,
    NativeTargetRunner,
)
from vllm.v1.worker.native_target_binding import (
    NativeClientBackend,
    RetainedNativeBinding,
)
from vllm.v1.worker.native_target_sampler import NativeVllmSampler


def config(slots=2, batch=4, max_context=64):
    return SimpleNamespace(
        additional_config={
            "afd_native_target": {"slots": slots, "batch_tokens": batch}
        },
        model_config=SimpleNamespace(
            max_model_len=max_context,
            logprobs_mode="raw_logprobs",
            use_fp64_gumbel=False,
            get_vocab_size=lambda: 8,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=slots, max_num_batched_tokens=batch * 2
        ),
    )


def request(name="a", prompt=(1, 2, 3), **kwargs):
    params = SamplingParams(
        max_tokens=kwargs.pop("max_tokens", 2),
        temperature=kwargs.pop("temperature", 0),
        **kwargs,
    )
    params.update_from_generation_config({"eos_token_id": 7})
    return Request(name, list(prompt), params, None)


class Bank:
    """Finite token credits for queue tests; no claim about native page geometry."""

    def __init__(self, capacity=128, batch=4):
        self.capacity = capacity
        self.batch = batch
        self.ends, self.generations = {}, {}
        self.events = []

    def info(self):
        free = self.capacity - sum(self.ends.values())
        return NativeCacheInfo(17, self.batch, (self.capacity,) * 4, (free,) * 4, 4096)

    def rpc(self, name, args=()):
        self.events.append((name, args))
        if name == "native_cache_info":
            value = self.info()
        elif name == "native_admit":
            slot, request_id = args
            self.generations[slot] = self.generations.get(slot, 0) + 1
            value = NativeRequest(17, slot, self.generations[slot])
            self.ends[value] = 0
        elif name == "native_can_prepare":
            (work,) = args
            value = (
                sum(self.ends.values()) + sum(tokens for _, tokens in work)
                <= self.capacity
            )
        elif name == "native_release":
            self.ends.pop(args[0])
            value = self.info()
        elif name == "native_cancel_step":
            value = self.info()
        else:
            raise AssertionError(name)
        return [value]


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.bank = Bank()
        self.scheduler = NativeScheduler(
            vllm_config=config(),
            executor=SimpleNamespace(collective_rpc=self.bank.rpc),
            cache_info=self.bank.info(),
            structured_output_manager=None,
            include_finished_set=True,
            log_stats=True,
        )

    def finish_step(self, work, token=5):
        ids, tokens, committed = [], [], []
        for grant in work.native_target.grants:
            end = grant.committed_end + len(grant.tokens)
            self.bank.ends[grant.request] = end
            ids.append(grant.request_id)
            current = self.scheduler.requests.get(grant.request_id)
            tokens.append([token] if current and end >= current.num_tokens else [])
            committed.append((grant.request_id, end))
        output = ModelRunnerOutput(ids, dict(zip(ids, range(len(ids)))), tokens)
        output.native_target = NativeStepResult(
            17, work.native_target.step_id, tuple(committed), self.bank.info()
        )
        return self.scheduler.update_from_output(work, output)

    def test_chunked_prefill_then_decode_preserves_usage_and_length_stop(self):
        req = request(prompt=tuple(range(7)), max_tokens=2)
        self.scheduler.add_request(req)
        first = self.scheduler.schedule()
        self.assertEqual(first.num_scheduled_tokens, {"a": 4})
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertFalse(self.finish_step(first).get(0).outputs)
        second = self.scheduler.schedule()
        self.assertEqual(second.num_scheduled_tokens, {"a": 3})
        result = self.finish_step(second)[0].outputs[0]
        self.assertEqual(result.new_token_ids, [5])
        self.assertEqual(result.prefill_stats.num_prompt_tokens, 7)
        self.assertEqual(result.prefill_stats.num_cached_tokens, 0)
        third = self.scheduler.schedule()
        self.assertEqual(third.native_target.grants[0].kind, "decode")
        result = self.finish_step(third)[0].outputs[0]
        self.assertTrue(result.finished)
        self.assertEqual(req.num_output_tokens, 2)
        self.assertFalse(self.bank.ends)

    def test_two_lanes_progress_and_third_request_queues_until_release(self):
        for name in "abc":
            self.scheduler.add_request(request(name, max_tokens=1))
        first = self.scheduler.schedule()
        self.assertEqual(set(first.num_scheduled_tokens), {"a", "b"})
        self.assertEqual(self.scheduler.get_request_counts(), (2, 1))
        self.finish_step(first)
        next_work = self.scheduler.schedule()
        self.assertEqual(set(next_work.num_scheduled_tokens), {"c"})
        self.finish_step(next_work)
        self.assertFalse(self.bank.ends)

    def test_native_future_capacity_denial_queues_without_partial_grant(self):
        self.bank.capacity = 8
        self.scheduler.cache_info = self.bank.info()
        self.scheduler.add_request(request("a", max_tokens=4))
        self.scheduler.add_request(request("b", max_tokens=4))
        work = self.scheduler.schedule()
        self.assertEqual(set(work.num_scheduled_tokens), {"a"})
        self.assertEqual(self.scheduler.get_request_counts(), (1, 1))
        self.assertEqual(len(self.bank.ends), 1)
        self.finish_step(work)
        self.scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
        work = self.scheduler.schedule()
        self.assertEqual(set(work.num_scheduled_tokens), {"b"})

    def test_unfit_request_reports_error_without_blocking_next_request(self):
        self.bank.capacity = 4
        self.scheduler.cache_info = self.bank.info()
        self.scheduler.add_request(request("large", max_tokens=8))
        self.scheduler.add_request(request("small", max_tokens=1))
        result = self.finish_step(self.scheduler.schedule())[0]
        self.assertEqual({out.request_id for out in result.outputs}, {"large", "small"})
        self.assertTrue(all(out.finished for out in result.outputs))
        self.assertFalse(self.bank.ends)

    def test_abort_between_execute_and_update_discards_tokens_and_frees_actual_bank(
        self,
    ):
        req = request()
        self.scheduler.add_request(req)
        work = self.scheduler.schedule()
        for grant in work.native_target.grants:
            self.bank.ends[grant.request] = len(grant.tokens)
        self.scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
        output = ModelRunnerOutput(["a"], {"a": 0}, [[5]])
        output.native_target = NativeStepResult(17, 1, (("a", 3),), self.bank.info())
        self.scheduler.update_from_output(work, output)
        self.assertFalse(req.output_token_ids)
        self.assertFalse(self.bank.ends)

    def test_reset_recomputes_existing_output_without_emitting_it_twice(self):
        req = request(max_tokens=3)
        self.scheduler.add_request(req)
        self.finish_step(self.scheduler.schedule())
        old = next(iter(self.bank.ends))
        self.assertTrue(self.scheduler.reset_prefix_cache(reset_running_requests=True))
        self.assertEqual(req.num_computed_tokens, 0)
        work = self.scheduler.schedule()
        self.assertNotEqual(work.native_target.grants[0].request, old)
        self.assertEqual(work.native_target.grants[0].tokens, (1, 2, 3, 5))
        self.finish_step(work)
        self.assertEqual(list(req.output_token_ids), [5, 5])

    def test_stale_or_wrong_extent_receipt_cannot_advance_request(self):
        req = request()
        self.scheduler.add_request(req)
        work = self.scheduler.schedule()
        output = ModelRunnerOutput(["a"], {"a": 0}, [[5]])
        output.native_target = NativeStepResult(18, 1, (("a", 3),), self.bank.info())
        with self.assertRaises(RuntimeError):
            self.scheduler.update_from_output(work, output)
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertFalse(req.output_token_ids)

    def test_invalid_second_row_cannot_publish_first_request(self):
        for name in "ab":
            self.scheduler.add_request(request(name))
        work = self.scheduler.schedule()
        output = ModelRunnerOutput(["a", "b"], {"a": 0, "b": 1}, [[5], []])
        output.native_target = NativeStepResult(
            17, 1, (("a", 3), ("b", 3)), self.bank.info()
        )
        with self.assertRaises(RuntimeError):
            self.scheduler.update_from_output(work, output)
        for req in self.scheduler.requests.values():
            self.assertEqual(req.num_computed_tokens, 0)
            self.assertFalse(req.output_token_ids)

    def test_release_timeout_retains_logical_owner_for_retry(self):
        self.scheduler.add_request(request())
        self.scheduler.schedule()
        rpc = self.bank.rpc

        def failed_release(name, args=()):
            if name == "native_release":
                raise TimeoutError("native release ACK pending")
            return rpc(name, args)

        self.scheduler.executor.collective_rpc = failed_release
        with self.assertRaises(TimeoutError):
            self.scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
        self.assertIn("a", self.scheduler.leases)
        self.scheduler.executor.collective_rpc = rpc
        self.scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
        self.assertFalse(self.bank.ends)

    def test_unsupported_request_rejected_before_native_admission(self):
        req = request(prompt_logprobs=1)
        with self.assertRaises(ValueError):
            self.scheduler.add_request(req)
        self.assertFalse(self.bank.ends)

    def test_eos_and_explicit_stop_token_use_vllm_stop_rules(self):
        req = request(stop_token_ids=[5], max_tokens=8)
        self.scheduler.add_request(req)
        result = self.finish_step(self.scheduler.schedule())[0].outputs[0]
        self.assertEqual(result.stop_reason, 5)
        self.assertTrue(result.finished)


class StreamingSchedulerTests(unittest.TestCase):
    finish_step = SchedulerTests.finish_step

    def setUp(self):
        self.bank = Bank(capacity=2048, batch=80)
        cfg = config(batch=80, max_context=1024)
        cfg.additional_config["afd_native_target"]["prefill_mode"] = "encoder_stream"
        self.scheduler = NativeScheduler(
            cfg,
            SimpleNamespace(collective_rpc=self.bank.rpc),
            self.bank.info(),
            None,
        )

    def test_whole_prompt_envelope_is_distinct_from_internal_chunk_budget(self):
        req = request(prompt=tuple(i % 7 for i in range(370)))
        self.scheduler.add_request(req)
        work = self.scheduler.schedule()
        (grant,) = work.native_target.grants
        self.assertEqual(grant.phase, "encoder_stream")
        self.assertEqual(grant.chunk_rows, 80)
        self.assertEqual(grant.selected, (369,))
        self.assertEqual(work.total_num_scheduled_tokens, 370)
        self.assertGreater(370, self.scheduler.token_budget)
        self.assertEqual(req.num_computed_tokens, 0)
        result = self.finish_step(work)[0].outputs[0]
        self.assertEqual(req.num_computed_tokens, 370)
        self.assertEqual(result.prefill_stats.num_prompt_tokens, 370)
        self.assertEqual(result.prefill_stats.num_cached_tokens, 0)
        self.assertEqual(result.new_token_ids, [5])

    def test_two_requests_prefill_exclusively_then_both_decode(self):
        for name in "ab":
            self.scheduler.add_request(request(name))
        first = self.scheduler.schedule()
        self.assertEqual(list(first.num_scheduled_tokens), ["a"])
        self.finish_step(first)
        second = self.scheduler.schedule()
        self.assertEqual(list(second.num_scheduled_tokens), ["b"])
        self.assertEqual(second.native_target.grants[0].phase, "encoder_stream")
        self.finish_step(second)
        third = self.scheduler.schedule()
        self.assertEqual(list(third.num_scheduled_tokens), ["a", "b"])
        self.assertTrue(
            all(g.phase == "full_target" for g in third.native_target.grants)
        )
        self.finish_step(third)
        self.assertFalse(self.bank.ends)

    def test_decode_round_cannot_accidentally_share_context_with_fresh_stream(self):
        self.scheduler.add_request(request("a", max_tokens=4))
        self.finish_step(self.scheduler.schedule())
        self.scheduler.add_request(request("b"))
        decode = self.scheduler.schedule()
        self.assertEqual(list(decode.num_scheduled_tokens), ["a"])
        self.finish_step(decode)
        fresh = self.scheduler.schedule()
        self.assertEqual(list(fresh.num_scheduled_tokens), ["b"])
        self.assertEqual(fresh.native_target.grants[0].phase, "encoder_stream")

    def test_recompute_with_existing_output_uses_bounded_full_target_grants(self):
        req = request(prompt=tuple(i % 7 for i in range(370)), max_tokens=4)
        self.scheduler.add_request(req)
        self.finish_step(self.scheduler.schedule())
        self.scheduler.reset_prefix_cache(reset_running_requests=True)
        work = self.scheduler.schedule()
        (grant,) = work.native_target.grants
        self.assertEqual(grant.phase, "full_target")
        self.assertEqual(len(grant.tokens), 80)
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertFalse(
            self.finish_step(work).get(0, SimpleNamespace(outputs=[])).outputs
        )


class SamplerTests(unittest.TestCase):
    def setUp(self):
        # Avoid invoking a compiler in this bounded CPU gate; the underlying
        # vLLM tensor function and sampler implementation still execute.
        import vllm.utils.torch_utils as torch_utils
        import vllm.v1.sample.ops.penalties as penalties
        import vllm.v1.sample.sampler as implementation

        # This CUDA wheel runs here without a driver. Only disable page locking
        # for CPU test allocations; production continues using vLLM's flags.
        for module in (implementation, penalties, torch_utils):
            pin_patch = patch.object(module, "PIN_MEMORY", False)
            pin_patch.start()
            self.addCleanup(pin_patch.stop)

        original = implementation.batched_count_greater_than
        self.count_patch = patch.object(
            implementation,
            "batched_count_greater_than",
            getattr(original, "_torchdynamo_orig_callable", original),
        )
        self.count_patch.start()
        self.addCleanup(self.count_patch.stop)
        self.sampler = NativeVllmSampler(config(), torch.device("cpu"))

    def work(self, req, tokens=None, end=0):
        tokens = tuple(req.prompt_token_ids) if tokens is None else tokens
        grant = NativeGrant(
            req.request_id,
            NativeRequest(17, 0, 1),
            0,
            end,
            tokens,
            (len(tokens) - 1,),
            "prefill",
            0,
        )
        output = SchedulerOutput.make_empty()
        output.scheduled_new_reqs = [NewRequestData.from_request(req, ())]
        output.native_target = NativeSchedule(17, 1, (grant,))
        self.sampler.update_requests(output)
        return grant

    def test_greedy_logprobs_and_readonly_native_logits(self):
        req = request(logprobs=3)
        grant = self.work(req)
        logits = torch.tensor([[0.0, 1.0, 3.0, 2.0, -2.0, -1.0, 0.5, 0.0]])
        before = logits.clone()
        result = self.sampler.sample((grant,), (SimpleNamespace(tensor=logits),), None)
        self.assertEqual(result.output.sampled_token_ids, [[2]])
        self.assertTrue(torch.equal(logits, before))
        expected = logits.log_softmax(-1)[0, 2].item()
        self.assertAlmostEqual(
            result.output.logprobs.logprobs[0, 0], expected, places=6
        )

    def test_min_tokens_and_logit_bias_use_existing_processors(self):
        req = request(
            min_tokens=1, max_tokens=3, stop_token_ids=[5], logit_bias={2: 10.0}
        )
        grant = self.work(req)
        logits = torch.tensor([[0.0, 1.0, 3.0, 2.0, -2.0, 100.0, 0.5, 0.0]])
        result = self.sampler.sample((grant,), (SimpleNamespace(tensor=logits),), None)
        self.assertEqual(result.output.sampled_token_ids, [[2]])

    def test_partial_prefill_does_not_consume_rng_or_emit_token(self):
        req = request(prompt=(1, 2, 3, 4, 5))
        grant = self.work(req, tokens=(1, 2, 3))
        result = self.sampler.sample(
            (grant,), (SimpleNamespace(tensor=torch.ones(1, 8)),), None
        )
        self.assertEqual(result.output.sampled_token_ids, [[]])
        self.assertEqual(result.accepted, (3,))

    def test_seeded_sampling_is_independent_of_interleaved_request(self):
        logits = torch.tensor([[0.0, 1.0, 3.0, 2.0, -2.0, -1.0, 0.5, 0.0]])
        a = self.work(request("a", temperature=0.8, seed=123, top_k=4, top_p=0.9))
        b = self.work(request("b", temperature=0.8, seed=123, top_k=4, top_p=0.9))
        sequence = {"a": [], "b": []}
        for index in range(5):
            for grant in (a, b):
                grant = replace(grant, committed_end=index)
                result = self.sampler.sample(
                    (grant,), (SimpleNamespace(tensor=logits),), None
                )
                sequence[grant.request_id].extend(result.output.sampled_token_ids[0])
        self.assertEqual(sequence["a"], sequence["b"])
        self.assertEqual(len(sequence["a"]), 5)

    def test_allowed_tokens_and_repetition_penalty_use_upstream_sampler(self):
        req = request(repetition_penalty=2.0, allowed_token_ids=[2, 4])
        grant = self.work(req)
        logits = torch.tensor([[0.0, 100.0, 5.0, 2.0, 3.0, 0.0, 0.5, 0.0]])
        result = self.sampler.sample((grant,), (SimpleNamespace(tensor=logits),), None)
        self.assertEqual(result.output.sampled_token_ids, [[4]])

    def test_identical_logits_keep_top_logprobs_with_mixed_prefill_and_decode(self):
        logits = torch.tensor([[0.0, 1.0, 3.0, 2.0, -2.0, -1.0, 0.5, 0.0]])
        a = self.work(request("a", logprobs=3))
        b = self.work(request("b", prompt=(1, 2, 3, 4, 5), logprobs=1), (1, 2))
        first = self.sampler.sample((a,), (SimpleNamespace(tensor=logits),), None)
        second = self.sampler.sample(
            (b, replace(a, committed_end=1)),
            (SimpleNamespace(tensor=logits), SimpleNamespace(tensor=logits)),
            None,
        )
        self.assertEqual(second.output.sampled_token_ids, [[], [2]])
        self.assertEqual(second.output.logprobs.cu_num_generated_tokens, [0, 0, 1])
        self.assertTrue(
            (first.output.logprobs.logprobs == second.output.logprobs.logprobs).all()
        )

    def test_scheduler_runner_sampler_two_context_lifecycle(self):
        self._native_flow("full_target")

    def test_stream_scheduler_runner_sampler_replay_publication_and_queue(self):
        self._native_flow("encoder_stream")

    def _native_flow(self, mode, dspark=False):
        streaming = mode == "encoder_stream"
        cfg = config(batch=80 if streaming else 4, max_context=1024)
        cfg.additional_config["afd_native_target"]["prefill_mode"] = mode
        if dspark:
            cfg.additional_config["afd_native_target"]["dspark"] = {
                "draft_limit": 3,
                "adaptive": False,
                "confidence_cutoff": None,
            }
        sampler = NativeVllmSampler(cfg, torch.device("cpu"))
        bank = Bank(capacity=2048, batch=80 if streaming else 4)
        events, tickets = [], {}

        def submit(grant):
            if grant.phase == "encoder_stream":
                self.assertFalse(tickets)
                self.assertEqual(bank.ends[grant.request], 0)
            tickets[grant.request_id] = grant
            events.append(("submit", grant.request_id))
            return grant.request_id

        def acquire(ticket):
            rows = (
                len(tickets[ticket].tokens) if tickets[ticket].phase == "dspark" else 1
            )
            return SimpleNamespace(
                tensor=torch.tensor(
                    [[0.0, 1.0, 3.0, 2.0, -2.0, -1.0, 0.5, 0.0]]
                ).repeat(rows, 1),
                drain_consumers=lambda: events.append(("drain", ticket)),
                close=lambda: events.append(("close", ticket)),
            )

        def commit(ticket, accepted):
            self.assertIn(("close", ticket), events)
            grant = tickets.pop(ticket)
            events.append(("commit", ticket))
            bank.ends[grant.request] += accepted
            return bank.ends[grant.request]

        def propose(grant):
            tokens = (*grant.tokens, *((2,) * (grant.verification_rows - 1)))
            prepared = replace(grant, tokens=tokens, selected=tuple(range(len(tokens))))
            return NativeProposal(submit(prepared), tokens, 1)

        backend = SimpleNamespace(
            info=bank.info,
            admit=lambda slot, seq: bank.rpc("native_admit", (slot, seq))[0],
            can_prepare=lambda work: bank.rpc("native_can_prepare", (work,))[0],
            committed_end=lambda req: bank.ends[req],
            submit=submit,
            propose=propose,
            execute=lambda work: events.append(("execute", work)),
            acquire_logits=acquire,
            commit=commit,
            cancel=lambda ticket: tickets.pop(ticket),
            release=lambda req: bank.rpc("native_release", (req,)),
            close=lambda: None,
        )
        runner = NativeTargetRunner(
            cfg,
            "cpu",
            SimpleNamespace(
                create_sampler=lambda *a: sampler,
                create_backend=lambda *a: backend,
            ),
            owner=17,
        )
        runner.load_model()
        self.addCleanup(runner.shutdown)

        def rpc(name, args=()):
            if name == "native_release":
                runner.release(*args)
                return [runner.cache_info()]
            method = {
                "native_cache_info": "cache_info",
                "native_admit": "admit",
                "native_can_prepare": "can_prepare",
            }[name]
            return [getattr(runner, method)(*args)]

        scheduler = NativeScheduler(
            cfg, SimpleNamespace(collective_rpc=rpc), bank.info(), None
        )
        for name in "abc":
            prompt = tuple(i % 7 for i in range(370)) if streaming else (1, 2, 3)
            scheduler.add_request(
                request(name, prompt, max_tokens=5 if dspark else 2, logprobs=3)
            )
        outputs = []
        for _ in range(10):
            work = scheduler.schedule()
            result = runner.execute_model(work)
            if result is None:
                result = runner.sample_tokens(None)
            for output in scheduler.update_from_output(work, result).values():
                outputs.extend(output.outputs)
            if not scheduler.has_requests():
                break
        self.assertEqual(len(outputs), 6)
        self.assertTrue(
            all(item.new_token_ids == [2] * len(item.new_token_ids) for item in outputs)
        )
        if dspark:
            self.assertEqual(sum(len(item.new_token_ids) for item in outputs), 15)
            self.assertTrue(any(len(item.new_token_ids) == 4 for item in outputs))
        self.assertEqual(sum(item.finished for item in outputs), 3)
        self.assertFalse(bank.ends)
        self.assertFalse(tickets)
        if streaming:
            self.assertEqual(events[:2], [("submit", "a"), ("execute", ("a",))])
            self.assertEqual([item.request_id for item in outputs], list("ababcc"))
        else:
            self.assertEqual(
                events[:3], [("submit", "a"), ("submit", "b"), ("execute", ("a", "b"))]
            )


class ClientMappingTests(unittest.TestCase):
    def test_proposal_preserves_native_ticket_and_owned_tokens(self):
        ticket = object()
        client = SimpleNamespace(
            submit_speculative=lambda grant: SimpleNamespace(
                ticket=ticket, tokens=(1, 2), draft_us=9
            )
        )
        proposal = NativeClientBackend(client).propose(object())
        self.assertEqual(proposal, NativeProposal(ticket, (1, 2), 9))

    def test_actual_cache_total_is_preserved_independently_of_payload_budget(self):
        info = SimpleNamespace(
            owner=17,
            capacity_rows=4,
            source_page_capacity=(4,) * 4,
            source_pages_free=(3,) * 4,
            cache_bytes=9999,
            source_payload_bytes=4096,
        )
        backend = NativeClientBackend(SimpleNamespace(info=lambda: info))
        self.assertEqual(backend.info().cache_bytes, 9999)

    def test_stream_cancel_reports_native_revocation_without_second_release(self):
        request = SimpleNamespace(owner=17, slot=0, generation=1)
        ticket = SimpleNamespace(request=request)
        client = SimpleNamespace(
            pending_commands=(), active_tickets=(ticket,), active_requests=(request,)
        )

        def cancel(value):
            self.assertIs(value, ticket)
            client.active_tickets = client.active_requests = ()

        client.cancel = cancel
        self.assertTrue(NativeClientBackend(client).cancel(ticket))

    def test_lost_commit_ack_recovery_does_not_report_request_revoked(self):
        request = SimpleNamespace(owner=17, slot=0, generation=1)
        ticket = SimpleNamespace(request=request)
        client = SimpleNamespace(
            pending_commands=(7,), active_tickets=(ticket,), active_requests=(request,)
        )

        def wait(command):
            self.assertEqual(command, 7)
            client.active_tickets = ()

        client.wait = wait
        self.assertFalse(NativeClientBackend(client).cancel(ticket))


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        lib = Path(self.directory.name) / "native.so"
        lib.write_bytes(b"test only")
        self.config = cfg = config()
        cfg.additional_config["afd_native_target"].update(
            enabled=True,
            implementation="retained",
            abi_library=str(lib),
            native_lib=str(lib),
            snapshot=self.directory.name,
            peers=[f"127.0.0.{rank + 1}:9200" for rank in range(4)],
            batch_tokens=80,
            source_pool_budget_bytes=4096,
        )
        cfg.model_config.get_vocab_size = lambda: 129280
        cfg.model_config.hf_config = SimpleNamespace(
            hidden_size=5120, num_hidden_layers=40
        )
        cfg.model_config.enable_return_routed_experts = False
        cfg.model_config.return_sampling_mask = False
        cfg.cache_config = SimpleNamespace(kv_cache_memory_bytes=None)
        client = ModuleType("vllm_afd.native_target.client")
        client.NativeTargetClient = SimpleNamespace(can_prepare=lambda *a: True)
        dependency = patch.dict("sys.modules", {client.__name__: client})
        dependency.start()
        self.addCleanup(dependency.stop)
        self.binding = RetainedNativeBinding()

    def test_supported_configuration_validates_without_creating_native_owner(self):
        self.binding.validate_config(self.config)

    def test_speculative_configuration_requires_typed_limits_and_matching_client(self):
        options = self.config.additional_config["afd_native_target"]
        good = {"draft_limit": 3, "adaptive": False, "confidence_cutoff": None}
        for changed in (
            {"draft_limit": 0},
            {"draft_limit": True},
            {"adaptive": 1},
            {"confidence_cutoff": float("nan")},
            {"unexpected": True},
        ):
            options["dspark"] = {**good, **changed}
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.binding.validate_config(self.config)
        options["dspark"] = good
        with self.assertRaisesRegex(RuntimeError, "lacks dSpark"):
            self.binding.validate_config(self.config)
        import sys

        client = sys.modules["vllm_afd.native_target.client"].NativeTargetClient
        client.submit_speculative = lambda grant: None
        self.binding.validate_config(self.config)

    def test_wrong_checkpoint_geometry_rejected_before_construction(self):
        self.config.model_config.hf_config.hidden_size = 4096
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.binding.validate_config(self.config)

    def test_encoder_stream_rejects_smaller_than_native_minimum_chunk_budget(self):
        options = self.config.additional_config["afd_native_target"]
        options["prefill_mode"] = "encoder_stream"
        self.config.scheduler_config.max_num_batched_tokens = 79
        with self.assertRaisesRegex(ValueError, "chunk budget"):
            self.binding.validate_config(self.config)

    def test_ordinary_kv_budget_or_sampling_mask_is_not_silently_ignored(self):
        self.config.cache_config.kv_cache_memory_bytes = 1234
        with self.assertRaises(ValueError):
            self.binding.validate_config(self.config)
        self.config.cache_config.kv_cache_memory_bytes = None
        self.config.model_config.return_sampling_mask = True
        with self.assertRaises(ValueError):
            self.binding.validate_config(self.config)


class SpeculativeSchedulerTests(unittest.TestCase):
    setUp = SchedulerTests.setUp
    finish_step = SchedulerTests.finish_step

    def prepare(self, **kwargs):
        self.scheduler.draft_limit = 3
        req = request(max_tokens=kwargs.pop("max_tokens", 8), **kwargs)
        self.scheduler.add_request(req)
        self.finish_step(self.scheduler.schedule())
        return req

    def test_verification_envelope_uses_native_credits_then_only_actual_end_advances(
        self,
    ):
        req = self.prepare()
        work = self.scheduler.schedule()
        (grant,) = work.native_target.grants
        self.assertEqual(
            (grant.phase, grant.tokens, grant.verification_rows), ("dspark", (5,), 4)
        )
        self.assertEqual(work.num_scheduled_tokens, {"a": 4})
        self.assertEqual(
            self.bank.events[-1], ("native_can_prepare", (((grant.request, 4),),))
        )
        self.bank.ends[grant.request] += 2
        output = ModelRunnerOutput(["a"], {"a": 0}, [[2, 4]])
        output.native_target = NativeStepResult(
            17,
            work.native_target.step_id,
            (("a", 5),),
            self.bank.info(),
            (("a", 2, 1),),
        )
        result = self.scheduler.update_from_output(work, output)[0]
        self.assertEqual(req.num_computed_tokens, 5)
        self.assertEqual(list(req.output_token_ids), [5, 2, 4])
        stats = result.scheduler_stats.spec_decoding_stats
        self.assertEqual((stats.num_draft_tokens, stats.num_accepted_tokens), (2, 1))
        next_work = self.scheduler.schedule()
        self.assertEqual(next_work.native_target.grants[0].tokens, (4,))
        self.assertEqual(next_work.native_target.grants[0].committed_end, 5)

    def test_output_and_context_limits_bound_anchor_only_fallback(self):
        req = self.prepare(max_tokens=2)
        work = self.scheduler.schedule()
        self.assertEqual(work.native_target.grants[0].verification_rows, 1)
        self.assertEqual(req.num_computed_tokens, 3)

    def test_context_space_includes_the_replacement_token(self):
        self.scheduler.max_model_len = 5
        self.prepare()
        self.assertEqual(
            self.scheduler.schedule().native_target.grants[0].verification_rows, 1
        )

    def test_sampling_options_missing_from_upstream_rejection_use_target_only(self):
        for option in ({"logit_bias": {2: 3.0}}, {"min_p": 0.1, "temperature": 0.8}):
            with self.subTest(option=option):
                self.setUp()
                self.prepare(**option)
                grant = self.scheduler.schedule().native_target.grants[0]
                self.assertEqual(
                    (grant.phase, grant.verification_rows), ("full_target", 0)
                )

    def test_bad_accepted_count_cannot_publish_any_output(self):
        req = self.prepare()
        work = self.scheduler.schedule()
        output = ModelRunnerOutput(["a"], {"a": 0}, [[2]])
        output.native_target = NativeStepResult(
            17,
            work.native_target.step_id,
            (("a", 5),),
            self.bank.info(),
            (("a", 2, 1),),
        )
        with self.assertRaises(RuntimeError):
            self.scheduler.update_from_output(work, output)
        self.assertEqual(
            (req.num_computed_tokens, list(req.output_token_ids)), (3, [5])
        )

    def test_native_metrics_count_verified_drafts_without_ordinary_draft_config(self):
        from functools import partial

        from prometheus_client import CollectorRegistry, Counter

        from vllm.v1.spec_decode.metrics import SpecDecodingProm, SpecDecodingStats

        registry = CollectorRegistry()
        with patch.object(
            SpecDecodingProm, "_counter_cls", partial(Counter, registry=registry)
        ):
            metrics = SpecDecodingProm(
                None,
                ["model", "engine"],
                {0: ["native", "0"]},
                native_num_speculative_tokens=3,
            )
        stats = SpecDecodingStats.new(3)
        stats.observe_draft(2, 1)
        metrics.observe(stats, 0)
        samples = {
            sample.name: sample.value
            for metric in registry.collect()
            for sample in metric.samples
        }
        self.assertEqual(samples["vllm:spec_decode_num_draft_tokens_total"], 2)
        self.assertEqual(samples["vllm:spec_decode_num_accepted_tokens_total"], 1)


class _CpuGreedyKernel:
    """CPU stub for the GPU primitive; real RejectionSampler orchestrates it."""

    def __getitem__(self, grid):
        return self.run

    def run(
        self,
        output,
        ends,
        drafts,
        argmax,
        bonus,
        greedy,
        limit,
        uniform,
        rates,
        **kwargs,
    ):
        assert greedy is None and not kwargs["SYNTHETIC_MODE"]
        start = 0
        for row, end in enumerate(ends.tolist()):
            for index in range(start, end):
                output[row, index - start] = argmax[index]
                if argmax[index] != drafts[index]:
                    break
            else:
                output[row, end - start] = bonus[row, 0]
            start = end


class _CpuExpandKernel(_CpuGreedyKernel):
    def run(self, output, values, ends, replace_from, replace_to, **kwargs):
        start = 0
        for value, end in zip(values, ends.tolist()):
            output[start:end] = replace_to if value == replace_from else value
            start = end


class _CpuRecoveredKernel(_CpuGreedyKernel):
    def run(
        self, output, ends, drafts, draft_probs, target, inv_q, vocab, block, **kwargs
    ):
        assert draft_probs is None and kwargs["NO_DRAFT_PROBS"]
        start = 0
        for row, end in enumerate(ends.tolist()):
            for index in range(start, end):
                probabilities = target[index].clone()
                probabilities[drafts[index]] = 0
                output[index] = (probabilities * inv_q[row]).argmax()
            start = end


class _CpuRandomKernel(_CpuGreedyKernel):
    def run(
        self,
        output,
        ends,
        drafts,
        draft_probs,
        target,
        bonus,
        recovered,
        uniform,
        greedy,
        limit,
        vocab,
        rates,
        **kwargs,
    ):
        assert draft_probs is None and kwargs["NO_DRAFT_PROBS"]
        start = 0
        for row, end in enumerate(ends.tolist()):
            for index in range(start, end):
                if uniform[index] >= target[index, drafts[index]]:
                    output[row, index - start] = recovered[index]
                    break
                output[row, index - start] = drafts[index]
            else:
                output[row, end - start] = bonus[row, 0]
            start = end


class SpeculativeSamplerTests(unittest.TestCase):
    def setUp(self):
        SamplerTests.setUp(self)
        import vllm.v1.sample.rejection_sampler as rejection

        for name, implementation in (
            ("rejection_greedy_sample_kernel", _CpuGreedyKernel()),
            ("rejection_random_sample_kernel", _CpuRandomKernel()),
            ("sample_recovered_tokens_kernel", _CpuRecoveredKernel()),
            ("expand_kernel", _CpuExpandKernel()),
        ):
            kernel = patch.object(rejection, name, implementation)
            kernel.start()
            self.addCleanup(kernel.stop)
        cfg = config(batch=4)
        cfg.additional_config["afd_native_target"]["dspark"] = {
            "draft_limit": 3,
            "adaptive": False,
            "confidence_cutoff": None,
        }
        self.sampler = NativeVllmSampler(cfg, "cpu")

    def test_two_context_native_proposals_publish_and_release_through_actual_vllm(self):
        SamplerTests._native_flow(self, "full_target", dspark=True)

    def test_stream_prefill_then_native_speculation_preserves_queue_progress(self):
        SamplerTests._native_flow(self, "encoder_stream", dspark=True)

    def verify(self, drafts=(2, 3), winners=(2, 3, 4), **params):
        req = request(max_tokens=params.pop("max_tokens", 8), logprobs=3, **params)
        req.append_output_token_ids(5)  # Already emitted, not yet computed anchor.
        new = SchedulerOutput.make_empty()
        new.scheduled_new_reqs = [
            NewRequestData.from_request(
                req, (), prefill_token_ids=list(req.all_token_ids)
            )
        ]
        self.sampler.update_requests(new)
        grant = NativeGrant(
            "a",
            NativeRequest(17, 0, 1),
            0,
            3,
            (5, *drafts),
            tuple(range(len(drafts) + 1)),
            "decode",
            0,
            phase="dspark",
            verification_rows=len(drafts) + 1,
        )
        logits = torch.zeros(len(winners), 8)
        for row, winner in enumerate(winners):
            logits[row, winner] = 4
        before = logits.clone()
        result = self.sampler.sample((grant,), (SimpleNamespace(tensor=logits),), None)
        self.assertTrue(torch.equal(before, logits))
        return result, logits

    def test_existing_rejection_sampler_accepts_prefix_and_bonus_with_row_logprobs(
        self,
    ):
        result, logits = self.verify()
        self.assertEqual(result.output.sampled_token_ids, [[2, 3, 4]])
        self.assertEqual(result.accepted, (3,))
        self.assertEqual(self.sampler.requests["a"].outputs, [5, 2, 3, 4])
        expected = logits.log_softmax(-1)[range(3), [2, 3, 4]]
        self.assertTrue(
            torch.allclose(
                torch.as_tensor(result.output.logprobs.logprobs[:, 0]), expected
            )
        )

    def test_first_and_middle_rejection_keep_replacement_without_suffix(self):
        for winners, expected in (((6, 3, 4), [6]), ((2, 6, 4), [2, 6])):
            with self.subTest(winners=winners):
                self.sampler.requests.clear()
                result, _ = self.verify(winners=winners)
                self.assertEqual(result.output.sampled_token_ids, [expected])
                self.assertEqual(result.accepted, (len(expected),))
                self.assertEqual(len(result.output.logprobs.logprobs), len(expected))

    def test_stop_and_output_limit_trim_before_native_commit(self):
        for params, expected in (
            ({"stop_token_ids": [2]}, [2]),
            ({"max_tokens": 3}, [2, 3]),
            ({}, [2, 7]),
        ):
            with self.subTest(params=params):
                self.sampler.requests.clear()
                winners = (2, 3, 4) if params else (2, 7, 4)
                drafts = (2, 3) if params else (2, 7)
                result, _ = self.verify(drafts, winners, **params)
                self.assertEqual(result.output.sampled_token_ids, [expected])
                self.assertEqual(result.accepted, (len(expected),))

    def test_anchor_only_proposal_uses_existing_sampler_without_zero_grid(self):
        result, _ = self.verify(drafts=(), winners=(6,))
        self.assertEqual(
            (result.output.sampled_token_ids, result.accepted), ([[6]], (1,))
        )

    def test_min_tokens_constraints_apply_at_each_verified_position(self):
        result, _ = self.verify(drafts=(7, 7), winners=(7, 7, 7), min_tokens=3)
        self.assertEqual(result.output.sampled_token_ids, [[0]])
        self.assertEqual(result.accepted, (1,))

    def test_random_target_sampling_uses_unit_draft_probability_and_target_constraints(
        self,
    ):
        for drafts, expected in (((2, 3), [2, 3, 4]), ((6, 3), [2])):
            with self.subTest(drafts=drafts):
                self.sampler.requests.clear()
                result, _ = self.verify(
                    drafts=drafts, temperature=0.8, top_k=1, seed=123
                )
                self.assertEqual(result.output.sampled_token_ids, [expected])
                self.assertEqual(result.accepted, (len(expected),))


if __name__ == "__main__":
    unittest.main(verbosity=2)
