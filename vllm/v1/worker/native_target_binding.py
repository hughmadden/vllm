# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production binding to the optional vllm-afd native target C ABI client."""

import ipaddress
import math
from pathlib import Path

from vllm.v1.worker.native_target import (
    NativeCacheInfo,
    NativeProposal,
    NativeRequest,
    NativeTargetUnavailable,
)


class NativeClientBackend:
    def __init__(self, client):
        # Retain the asynchronous native owner before waiting for initialization.
        self.client = client

    def initialize(self):
        self.client.initialize()

    def info(self):
        info = self.client.info()
        return NativeCacheInfo(
            info.owner,
            info.capacity_rows,
            tuple(info.source_page_capacity),
            tuple(info.source_pages_free),
            info.cache_bytes,
        )

    def admit(self, slot, request_id):
        request = self.client.admit(slot, request_id)
        return NativeRequest(request.owner, request.slot, request.generation)

    def can_prepare(self, work):
        return self.client.can_prepare(work)

    def committed_end(self, request):
        return self.client.committed_end(request)

    def submit(self, grant):
        return self.client.submit(grant)

    def propose(self, grant):
        proposal = self.client.submit_speculative(grant)
        return NativeProposal(
            proposal.ticket, tuple(proposal.tokens), proposal.draft_us
        )

    def execute(self, tickets):
        self.client.execute(tickets)

    def acquire_logits(self, ticket):
        return self.client.acquire_logits(ticket)

    def commit(self, ticket, accepted):
        return self.client.commit(ticket, accepted)

    def cancel(self, ticket):
        # A timed-out commit/submit still owns its original native command.
        # Recover its reply before deciding whether a ticket remains cancellable.
        for command in self.client.pending_commands:
            self.client.wait(command)
        if ticket in self.client.active_tickets:
            self.client.cancel(ticket)
        return ticket.request not in self.client.active_requests

    def release(self, request):
        self.client.release(request)

    def close(self):
        # The client recovers unresolved tickets before cancellation/release and
        # refuses to reclaim outstanding native logits consumer leases.
        self.client.close()


class RetainedNativeBinding:
    def validate_config(self, config):
        options = config.additional_config["afd_native_target"]
        required = {
            "enabled",
            "implementation",
            "abi_library",
            "snapshot",
            "native_lib",
            "peers",
            "batch_tokens",
            "slots",
            "source_pool_budget_bytes",
        }
        if not required <= options.keys() or options.keys() - required - {
            "timeout_s",
            "poll_interval_s",
            "prefill_mode",
            "dspark",
        }:
            raise ValueError(
                "native retained binding needs explicit artifacts, peers "
                "and memory limits"
            )
        if options["implementation"] != "retained":
            raise ValueError("unknown native target implementation")
        mode = options.get("prefill_mode", "full_target")
        if mode not in ("full_target", "encoder_stream"):
            raise ValueError("unknown native prefill mode")
        if mode == "encoder_stream" and (
            config.scheduler_config.max_num_batched_tokens < 80
        ):
            raise ValueError("native encoder stream requires an 80-row chunk budget")
        dspark = options.get("dspark")
        if dspark is not None:
            if (
                not isinstance(dspark, dict)
                or set(dspark) != {"draft_limit", "adaptive", "confidence_cutoff"}
                or type(dspark["draft_limit"]) is not int
                or not 1 <= dspark["draft_limit"] <= 5
                or type(dspark["adaptive"]) is not bool
                or options["slots"] < 2
            ):
                raise ValueError(
                    "native dSpark needs explicit proposal limits and two slots"
                )
            cutoff = dspark["confidence_cutoff"]
            if cutoff is not None and (
                type(cutoff) not in (int, float)
                or not math.isfinite(cutoff)
                or not 0 < cutoff < 1
            ):
                raise ValueError("invalid native draft confidence cutoff")
        for name in ("abi_library", "snapshot", "native_lib"):
            path = Path(options[name])
            if not path.is_absolute() or not path.exists():
                raise ValueError(
                    f"native {name} must name an existing absolute artifact"
                )
        if len(options["peers"]) != 4:
            raise ValueError("native target requires four numeric expert peers")
        for peer in options["peers"]:
            host, separator, port = peer.rpartition(":")
            if not separator or not port.isdigit() or not 0 < int(port) < 65536:
                raise ValueError("invalid native expert peer")
            ipaddress.ip_address(host.strip("[]"))
        for name, low, high in (
            ("slots", 1, 16),
            ("batch_tokens", 80, 4096),
            ("source_pool_budget_bytes", 1, 2**63 - 1),
        ):
            if type(options[name]) is not int or not low <= options[name] <= high:
                raise ValueError(f"invalid native {name}")
        if not 1 <= config.scheduler_config.max_num_seqs <= options["slots"]:
            raise ValueError("max_num_seqs must fit the explicit native request slots")
        model = config.model_config
        if (
            not 1 <= model.max_model_len <= 1048576
            or model.get_vocab_size() != 129280
            or model.hf_config.hidden_size != 5120
            or model.hf_config.num_hidden_layers != 40
        ):
            raise ValueError(
                "native target binding requires the retained DS4.1 Flash geometry"
            )
        if (
            model.enable_return_routed_experts
            or model.return_sampling_mask
            or getattr(model, "logits_processors", None)
            or config.cache_config.kv_cache_memory_bytes is not None
        ):
            raise ValueError(
                "native routed-expert capture, custom processors and "
                "vLLM KV budget are unbound"
            )
        try:
            from vllm_afd.native_target.client import NativeTargetClient
        except ImportError as exc:
            raise NativeTargetUnavailable(
                "install the matching retained vllm-afd target client"
            ) from exc
        if not hasattr(NativeTargetClient, "can_prepare"):
            raise NativeTargetUnavailable(
                "native target client lacks authoritative capacity queries"
            )
        if dspark is not None and not hasattr(NativeTargetClient, "submit_speculative"):
            raise NativeTargetUnavailable("native target client lacks dSpark proposals")

    def create_sampler(self, config, device):
        from vllm.v1.worker.native_target_sampler import NativeVllmSampler

        if device.type != "cuda" or device.index not in (None, 0):
            raise NativeTargetUnavailable(
                "retained native target requires CUDA device 0"
            )
        return NativeVllmSampler(config, device)

    def create_backend(self, config, device, owner):
        from vllm_afd.native_target.client import NativeTargetClient

        options = config.additional_config["afd_native_target"]
        target = {
            name: options[name]
            for name in (
                "snapshot",
                "native_lib",
                "peers",
                "batch_tokens",
                "slots",
                "source_pool_budget_bytes",
            )
        }
        target.update(owner=owner, max_context_tokens=config.model_config.max_model_len)
        if options.get("dspark") is not None:
            target["dspark"] = options["dspark"]
        return NativeClientBackend(
            NativeTargetClient(
                target,
                enabled=True,
                library_path=options["abi_library"],
                timeout_s=options.get("timeout_s", 60),
                poll_interval_s=options.get("poll_interval_s", 0.00005),
                device_id=0,
            )
        )

    def create_scheduler(self, **kwargs):
        from vllm.v1.core.sched.native_scheduler import NativeScheduler

        return NativeScheduler(**kwargs)
