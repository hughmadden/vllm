# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Existing vLLM sampling over leased native logits, with bounded GPU scratch."""

from dataclasses import dataclass

import torch

from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput
from vllm.v1.sample.logits_processor import BUILTIN_LOGITS_PROCESSORS
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.native_target import NativeSample


@dataclass
class _RequestSampling:
    params: object
    prompt: list[int]
    outputs: list[int]
    slot: int
    generator: torch.Generator | None


class NativeVllmSampler:
    def __init__(self, config, device):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.vocab = config.model_config.get_vocab_size()
        self.max_requests = config.scheduler_config.max_num_seqs
        self.max_context = config.model_config.max_model_len
        self.sampler = Sampler(
            logprobs_mode=config.model_config.logprobs_mode,
            use_fp64_gumbel=getattr(config.model_config, "use_fp64_gumbel", False),
        )
        self.processors = LogitsProcessors(
            cls(config, self.device, False) for cls in BUILTIN_LOGITS_PROCESSORS
        )
        # The native descriptor is read-only. vLLM processors mutate logits, so
        # consume the zero-copy view into reusable device scratch, never via CPU.
        self.logits = torch.empty((2, self.vocab), dtype=torch.float32, device=device)
        self.prompt_ids = torch.empty(
            (self.max_requests, self.max_context), dtype=torch.int64, device=device
        )
        self.allowed = torch.empty(
            (self.max_requests, self.vocab), dtype=torch.bool, device=device
        )
        self.scalars = torch.empty(
            (self.max_requests, 5), dtype=torch.float32, device=device
        )
        self.top_k = torch.empty(self.max_requests, dtype=torch.int32, device=device)
        self.requests = {}

    def update_requests(self, output):
        for name in output.finished_req_ids:
            self.requests.pop(name, None)
        for new in output.scheduled_new_reqs:
            prompt = list(new.prompt_token_ids)
            history = list((new.prefill_token_ids or prompt)[len(prompt) :])
            if new.req_id in self.requests:
                state = self.requests[new.req_id]
                if state.prompt != prompt or state.outputs != history:
                    raise RuntimeError("native resumed sampling history differs")
                continue
            used = {state.slot for state in self.requests.values()}
            slot = next(i for i in range(self.max_requests) if i not in used)
            params = new.sampling_params
            generator = None
            if params.seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(params.seed)
            state = _RequestSampling(params, prompt, history, slot, generator)
            self.requests[new.req_id] = state
            values = [
                params.temperature,
                params.top_p,
                params.frequency_penalty,
                params.presence_penalty,
                params.repetition_penalty,
            ]
            self.scalars[slot].copy_(
                torch.tensor(values, dtype=torch.float32), non_blocking=True
            )
            self.top_k[slot].fill_(
                self.vocab if params.top_k < 1 else min(params.top_k, self.vocab)
            )
            if (
                params.repetition_penalty != 1
                or params.frequency_penalty
                or params.presence_penalty
            ):
                self.prompt_ids[slot, : len(prompt)].copy_(
                    torch.tensor(prompt), non_blocking=True
                )
            if params.allowed_token_ids is not None:
                self.allowed[slot].fill_(True)
                self.allowed[slot, params.allowed_token_ids] = False

    def _metadata(self, state, max_logprobs):
        params, slot = state.params, state.slot
        greedy = params.temperature < 1e-5
        no_penalties = (
            params.frequency_penalty == 0
            and params.presence_penalty == 0
            and params.repetition_penalty == 1
        )
        # Reuse the upstream processors, resetting their single-row batch before
        # each independent request. Per-request RNG and output histories persist.
        removed = BatchUpdate(batch_size=0, removed=[0], added=[], moved=[])
        added = BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, params, state.prompt, state.outputs)],
            moved=[],
        )
        for processor in self.processors.all:
            processor.update_state(removed)
            processor.update_state(added)
        scalar = self.scalars[slot]
        return SamplingMetadata(
            temperature=scalar[0:1],
            all_greedy=greedy,
            all_random=not greedy,
            top_p=scalar[1:2],
            top_k=self.top_k[slot : slot + 1],
            generators={0: state.generator} if state.generator is not None else {},
            max_num_logprobs=max_logprobs,
            no_penalties=no_penalties,
            prompt_token_ids=(
                None
                if no_penalties
                else self.prompt_ids[slot : slot + 1, : len(state.prompt)]
            ),
            frequency_penalties=scalar[2:3],
            presence_penalties=scalar[3:4],
            repetition_penalties=scalar[4:5],
            output_token_ids=[state.outputs],
            allowed_token_ids_mask=(
                None
                if params.allowed_token_ids is None
                else self.allowed[slot : slot + 1]
            ),
            bad_words_token_ids={0: params.bad_words_token_ids}
            if params.bad_words_token_ids
            else {},
            logitsprocs=self.processors,
        )

    @torch.inference_mode()
    def sample(self, grants, leases, grammar_output):
        if grammar_output is not None:
            raise ValueError("native structured output is not bound")
        max_logprobs = max(
            (
                self.requests[g.request_id].params.logprobs
                for g in grants
                if self.requests[g.request_id].params.logprobs is not None
            ),
            default=None,
        )
        sampled, logprobs, boundaries = [], [], [0]
        for grant, lease in zip(grants, leases):
            state = self.requests[grant.request_id]
            if grant.committed_end + len(grant.tokens) < len(state.prompt) + len(
                state.outputs
            ):
                sampled.append([])
                boundaries.append(boundaries[-1])
                continue
            if grant.selected != (len(grant.tokens) - 1,):
                raise ValueError("native text sampler needs the last granted row")
            source = lease.tensor
            if tuple(source.shape) != (1, self.vocab) or source.dtype != torch.float32:
                raise RuntimeError("native selected logits extent differs")
            logits = self.logits[grant.lane : grant.lane + 1]
            logits.copy_(source, non_blocking=True)
            result = self.sampler(logits, self._metadata(state, max_logprobs))
            token = result.sampled_token_ids.item()
            state.outputs.append(token)
            sampled.append([token])
            boundaries.append(boundaries[-1] + 1)
            if result.logprobs_tensors is not None:
                logprobs.append(result.logprobs_tensors)
        names = [g.request_id for g in grants]
        output = ModelRunnerOutput(
            names,
            dict(zip(names, range(len(names)))),
            sampled,
            logprobs=(
                LogprobsTensors.cat(logprobs, boundaries).tolists()
                if logprobs
                else None
            ),
        )
        return NativeSample(output, tuple(len(g.tokens) for g in grants))

    def close(self):
        self.requests.clear()
        self.logits = self.prompt_ids = self.allowed = self.scalars = self.top_k = None
        self.sampler = self.processors = None
