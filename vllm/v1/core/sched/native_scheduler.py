# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text scheduler whose only physical cache owner is the retained native bank."""

from collections import defaultdict, deque

from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.stats import PrefillStats, SchedulerStats
from vllm.v1.request import RequestStatus
from vllm.v1.worker.native_target import (
    NativeCacheInfo,
    NativeGrant,
    NativeSchedule,
    NativeStepResult,
)


def validate_text_request(request, max_model_len):
    params = request.sampling_params
    if (
        params is None
        or request.pooling_params is not None
        or not request.prompt_token_ids
        or request.prompt_embeds is not None
        or request.mm_features
        or request.lora_request is not None
        or request.resumable
        or request.use_structured_output
        or request.priority != 0
        or params.prompt_logprobs is not None
        or params.logprobs == -1
        or params.logprob_token_ids
        or params.thinking_token_budget is not None
        or params.trace_decode_token_ids
        or params.extra_args
        or request.recurrent_instruction_boundary is not None
        or request.num_prompt_tokens >= max_model_len
    ):
        raise ValueError(
            "native target supports nonempty text generation with standard sampling "
            "and top-k output logprobs; image/embeds, structured output, priority, "
            "prompt/full-vocabulary logprobs and streaming input are not bound"
        )


class NativeScheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config,
        executor,
        cache_info,
        structured_output_manager,
        include_finished_set=False,
        log_stats=False,
    ):
        self.config, self.executor = vllm_config, executor
        self.max_model_len = vllm_config.model_config.max_model_len
        options = vllm_config.additional_config["afd_native_target"]
        self.max_requests = min(
            options["slots"], vllm_config.scheduler_config.max_num_seqs
        )
        self.batch_tokens = min(options["batch_tokens"], cache_info.capacity_rows)
        self.token_budget = vllm_config.scheduler_config.max_num_batched_tokens
        self.cache_info = cache_info
        self.owner = cache_info.owner
        self.log_stats, self.include_finished_set = log_stats, include_finished_set
        self.requests, self.leases = {}, {}
        self.running, self.waiting = [], deque()
        self._new, self._finished = set(), set()
        self._finished_by_client, self._errors = defaultdict(set), defaultdict(list)
        self._pause_state = PauseState.UNPAUSED
        self._inflight = None
        self._request_sequence = self.current_step = 0
        self.connector = self.ec_connector = None

    def _rpc(self, method, *args):
        replies = self.executor.collective_rpc(method, args=args)
        if len(replies) != 1:
            raise RuntimeError("native scheduler requires one worker reply")
        return replies[0]

    def _receipt(self, info):
        if not isinstance(info, NativeCacheInfo):
            raise RuntimeError("invalid native scheduler cache receipt")
        info.validate(self.owner)
        if info.source_page_capacity != self.cache_info.source_page_capacity:
            raise RuntimeError("native physical capacity changed within owner")
        self.cache_info = info

    def _can_prepare(self, work):
        if not work:
            return True
        result = self._rpc("native_can_prepare", tuple(work))
        if type(result) is not bool:
            raise RuntimeError("invalid native capacity check")
        return result

    def _remaining(self, request):
        end = min(
            self.max_model_len, request.num_prompt_tokens + request.max_tokens - 1
        )
        return max(0, end - request.num_computed_tokens)

    def _admit_waiting(self):
        if self._pause_state != PauseState.UNPAUSED:
            return
        while self.waiting and len(self.running) < self.max_requests:
            request = self.waiting[0]
            occupied = {lease.slot for lease in self.leases.values()}
            slot = next(i for i in range(self.max_requests) if i not in occupied)
            self._request_sequence += 1
            lease = self._rpc("native_admit", slot, self._request_sequence)
            work = [
                (self.leases[r.request_id], self._remaining(r))
                for r in self.running
                if self._remaining(r)
            ]
            candidate = (lease, self._remaining(request))
            if not self._can_prepare([*work, candidate]):
                fits_alone = self._can_prepare([candidate])
                self._receipt(self._rpc("native_release", lease))
                if fits_alone or self.running:
                    break
                self.waiting.popleft()
                request.status = RequestStatus.FINISHED_ERROR
                self._errors[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )
                self._retire(request)
                continue
            self.waiting.popleft()
            self.leases[request.request_id] = lease
            self.running.append(request)
            self._new.add(request.request_id)
            request.status = RequestStatus.RUNNING
            request.record_event(EngineCoreEventType.SCHEDULED)

    def add_request(self, request):
        self.validate_request(request)
        if request.request_id in self.requests:
            raise ValueError("native request ID already active")
        self.requests[request.request_id] = request
        self.waiting.append(request)
        request.record_event(EngineCoreEventType.QUEUED)
        request.prefill_stats = PrefillStats()
        request.prefill_stats.set(request.num_prompt_tokens, 0, 0)

    def validate_request(self, request):
        validate_text_request(request, self.max_model_len)

    def schedule(self, throttle_prefills=False):
        if self._inflight is not None:
            raise RuntimeError("native scheduler has an unacknowledged step")
        self.current_step += 1
        output = SchedulerOutput.make_empty()
        output.finished_req_ids, self._finished = self._finished, set()
        self._receipt(self._rpc("native_cache_info"))
        reusing_name = any(
            r.request_id in output.finished_req_ids for r in self.waiting
        )
        if not reusing_name:
            self._admit_waiting()
        grants = []
        remaining = self.token_budget
        if self._pause_state != PauseState.PAUSED_ALL and not reusing_name:
            decodes = [
                r for r in self.running if r.num_computed_tokens >= r.num_prompt_tokens
            ]
            candidates = self.running
            if throttle_prefills and decodes:
                candidates = decodes
            for request in candidates:
                if len(grants) == 2 or remaining == 0:
                    break
                start = request.num_computed_tokens
                count = min(request.num_tokens - start, self.batch_tokens, remaining)
                if count <= 0:
                    raise RuntimeError("native request has no granted input token")
                tokens = tuple(request.all_token_ids[start : start + count])
                grant = NativeGrant(
                    request.request_id,
                    self.leases[request.request_id],
                    len(grants),
                    start,
                    tokens,
                    (count - 1,),
                    "prefill" if start < request.num_prompt_tokens else "decode",
                    0,
                )
                grants.append(grant)
                output.num_scheduled_tokens[request.request_id] = count
                remaining -= count
                request.is_prefill_chunk = start + count < request.num_tokens
                if request.request_id in self._new:
                    output.scheduled_new_reqs.append(
                        NewRequestData.from_request(
                            request, (), prefill_token_ids=list(request.all_token_ids)
                        )
                    )
                    self._new.remove(request.request_id)
            if grants and not self._can_prepare(
                [(g.request, len(g.tokens)) for g in grants]
            ):
                raise RuntimeError("native capacity changed after logical admission")
            selected = {g.request_id for g in grants}
            self.running = [r for r in self.running if r.request_id not in selected] + [
                r for r in self.running if r.request_id in selected
            ]
        output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
        output.native_target = NativeSchedule(
            self.owner, self.current_step, tuple(grants)
        )
        self._inflight = output.native_target
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        work, receipt = (
            scheduler_output.native_target,
            model_runner_output.native_target,
        )
        expected = tuple(
            (g.request_id, g.committed_end + len(g.tokens)) for g in work.grants
        )
        if (
            work != self._inflight
            or not isinstance(receipt, NativeStepResult)
            or receipt.owner != self.owner
            or receipt.step_id != work.step_id
            or receipt.committed_ends != expected
            or model_runner_output.req_ids != [g.request_id for g in work.grants]
            or model_runner_output.req_id_to_index
            != {g.request_id: i for i, g in enumerate(work.grants)}
            or len(model_runner_output.sampled_token_ids) != len(work.grants)
        ):
            raise RuntimeError("invalid native publication acknowledgement")
        # Validate every row before publishing any request in this step.
        for index, grant in enumerate(work.grants):
            request = self.requests.get(grant.request_id)
            if request is None or self.leases.get(grant.request_id) != grant.request:
                continue
            end = grant.committed_end + len(grant.tokens)
            generated = model_runner_output.sampled_token_ids[index]
            if len(generated) != (1 if end >= request.num_tokens else 0):
                raise RuntimeError(
                    "native sampling count disagrees with prefill boundary"
                )
        self._receipt(receipt.cache_info)
        self._inflight = None
        outputs, self._errors = self._errors, defaultdict(list)
        for index, grant in enumerate(work.grants):
            request = self.requests.get(grant.request_id)
            if request is None or self.leases.get(grant.request_id) != grant.request:
                continue  # Aborted/preempted after GPU completion, before this ACK.
            end = grant.committed_end + len(grant.tokens)
            generated = model_runner_output.sampled_token_ids[index]
            request.num_computed_tokens = end
            for token in generated:
                request.append_output_token_ids(token)
                check_stop(request, self.max_model_len)
            if generated:
                logprobs = model_runner_output.logprobs
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=generated,
                        new_logprobs=(
                            logprobs.slice_request(index, len(generated))
                            if logprobs
                            and request.sampling_params.num_logprobs is not None
                            else None
                        ),
                        finish_reason=request.get_finished_reason(),
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=request.take_prefill_stats(),
                        trace_headers=request.trace_headers,
                    )
                )
            if request.is_finished():
                self._retire(request)
        # A release processed by an abort or stop is newer than the step receipt.
        self._receipt(self._rpc("native_cache_info"))
        result = {
            client: EngineCoreOutputs(outputs=items)
            for client, items in outputs.items()
        }
        for client, ids in self._finished_by_client.items():
            result.setdefault(client, EngineCoreOutputs()).finished_requests = ids
        self._finished_by_client = defaultdict(set)
        if self.log_stats:
            result.setdefault(
                0, EngineCoreOutputs()
            ).scheduler_stats = self.make_stats()
        return result

    def _retire(self, request):
        name = request.request_id
        if lease := self.leases.get(name):
            self._receipt(self._rpc("native_release", lease))
            self.leases.pop(name)
        self.requests.pop(name, None)
        self.running = [r for r in self.running if r.request_id != name]
        self.waiting = deque(r for r in self.waiting if r.request_id != name)
        self._new.discard(name)
        self._finished.add(name)
        if self.include_finished_set:
            self._finished_by_client[request.client_index].add(name)

    def finish_requests(self, request_ids, finished_status):
        ids = (
            list(self.requests)
            if request_ids is None
            else ([request_ids] if isinstance(request_ids, str) else list(request_ids))
        )
        finished = []
        for name in ids:
            if request := self.requests.get(name):
                request.status = finished_status
                self._retire(request)
                finished.append(request)
        return finished

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        if reset_connector:
            raise ValueError("native target has no KV connector")
        if reset_running_requests:
            for request in tuple(self.running):
                lease = self.leases[request.request_id]
                self._receipt(self._rpc("native_release", lease))
                self.leases.pop(request.request_id)
                request.num_computed_tokens = 0
                request.status = RequestStatus.PREEMPTED
                self.waiting.appendleft(request)
            self.running.clear()
        return True

    def get_num_unfinished_requests(self):
        return len(self.requests)

    def has_finished_requests(self):
        return bool(self._finished or self._errors)

    @property
    def pause_state(self):
        return self._pause_state

    def set_pause_state(self, pause_state):
        self._pause_state = PauseState(pause_state)

    def get_request_counts(self):
        return len(self.running), len(self.waiting)

    def get_kv_cache_usage(self):
        capacity = sum(self.cache_info.source_page_capacity)
        return (
            1 - sum(self.cache_info.source_pages_free) / capacity if capacity else 0.0
        )

    def make_stats(self):
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.get_kv_cache_usage(),
            step_counter=self.current_step,
        )

    def get_grammar_bitmask(self, scheduler_output):
        return None

    def update_draft_token_ids(self, draft_token_ids):
        raise ValueError("native speculation is not bound")

    def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output):
        raise ValueError("native speculation is not bound")

    def reset_encoder_cache(self):
        return None

    def get_prefill_fairness(self):
        return {"policy": "fcfs_native_two_lanes"}

    def set_prefill_fairness(self, config):
        raise ValueError("native scheduler fairness reconfiguration is not bound")

    def record_compute_time(self, service_class, elapsed_seconds, **kwargs):
        return None

    def shutdown(self):
        # EngineCore closes the physical worker first; only logical state remains.
        self.requests.clear()
        self.leases.clear()
        self.running.clear()
        self.waiting.clear()
