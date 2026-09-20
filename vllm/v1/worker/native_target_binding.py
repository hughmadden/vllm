# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production binding to the optional vllm-afd native target C ABI client."""

import ipaddress
import math
from pathlib import Path

from vllm.v1.worker.native_target import (
    NativeBatchMember,
    NativeBatchProposal,
    NativeCacheInfo,
    NativeCompositeRecovery,
    NativeProposal,
    NativeRequest,
    NativeTargetUnavailable,
)


class NativeClientBackend:
    def __init__(self, client, *, enable_composite=False):
        if type(enable_composite) is not bool:
            raise ValueError("composite opt-in must be explicit boolean")
        self._composite_enabled = enable_composite
        # Retain the asynchronous native owner before waiting for initialization.
        self.client = client
        self._composite_pending = {}
        self._composite_failures = {}

    def initialize(self):
        self.client.initialize()
        negotiate = getattr(self.client, "negotiate_composite", None)
        if self._composite_enabled and callable(negotiate):
            negotiate()

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

    def submit_batch(self, grant):
        return self.client.submit_batch(grant)

    def propose_batch(self, grant):
        proposal = self.client.submit_speculative_batch(grant)
        members = tuple(
            NativeBatchMember(
                NativeRequest(m.request.owner, m.request.slot, m.request.generation),
                tuple(m.tokens),
                tuple(m.selected),
                m.input_offset,
                m.output_offset,
            )
            for m in proposal.members
        )
        return NativeBatchProposal(proposal.ticket, members, proposal.draft_us)

    def can_drain_commit_batch(self, ticket, lease):
        supported = getattr(self.client, "composite_supported", None)
        return (
            self._composite_enabled
            and callable(supported) and supported() is True
            and self.client.active_tickets == (ticket,)
            and not self.client.pending_commands
            and lease.ticket == ticket
            and any(owned is lease for owned in self.client.active_leases)
            and not self._composite_failures
        )

    def drain_commit_batch(self, ticket, lease, accepted):
        from vllm_afd.errors import LeaseError
        if not self._composite_enabled:
            raise LeaseError("composite path is not explicitly enabled")
        from vllm_afd.native_target.composite import CompositeFailure
        from vllm_afd.native_target.client import NativeTargetTimeout

        try:
            return self.client.drain_commit_batch(lease, accepted)
        except NativeTargetTimeout as error:
            self._composite_pending[ticket] = error.command
            raise
        except CompositeFailure as error:
            self._composite_failures[ticket] = error
            raise

    def recover_composite(self, ticket):
        if not self._composite_pending and not self._composite_failures:
            return None
        if ticket not in self._composite_pending and ticket not in self._composite_failures:
            return None
        from vllm_afd.native_target.composite import CompositeFailure

        if ticket in self._composite_pending:
            command = self._composite_pending[ticket]
            try:
                self.client.wait(command)
            except CompositeFailure as error:
                del self._composite_pending[ticket]
                self._composite_failures[ticket] = error
            else:
                # A successful typed parser return already proved consumption.
                del self._composite_pending[ticket]
                return NativeCompositeRecovery(True)
        failure = self._composite_failures[ticket]
        outcome = failure.outcome
        if outcome.state == "quarantined":
            return NativeCompositeRecovery(outcome.consumer_complete, failure)
        del self._composite_failures[ticket]
        if outcome.state == "cancelled" and outcome.consumer_complete:
            return NativeCompositeRecovery(True)
        if outcome.state == "lease_held" and not outcome.consumer_complete:
            return NativeCompositeRecovery(False)
        raise RuntimeError("unsupported composite recovery proof")

    def commit_batch(self, ticket, accepted):
        return self.client.commit_batch(ticket, accepted)

    def execute(self, tickets):
        self.client.execute(tickets)

    def acquire_logits(self, ticket):
        return self.client.acquire_logits(ticket)

    def commit(self, ticket, accepted):
        return self.client.commit(ticket, accepted)

    def cancel(self, ticket):
        recovery = self.recover_composite(ticket)
        if recovery is not None and recovery.failure is not None:
            raise recovery.failure
        # A timed-out commit/submit still owns its original native command.
        # Recover its reply before deciding whether a ticket remains cancellable.
        for command in self.client.pending_commands:
            self.client.wait(command)
        if ticket in self.client.active_tickets:
            # Recovery can attach a logits lease to the client job after the
            # runner's acquisition timed out. Rediscover this ticket's ownership
            # and honor its consumer fence; leave other tickets' leases alone.
            # Keep ownership recoverable if draining or closing raises.
            lease = next(
                (lease for lease in getattr(self.client, "active_leases", ())
                 if lease.ticket == ticket),
                None,
            )
            if lease is not None:
                lease.drain_consumers()
                lease.close()
            self.client.cancel(ticket)
        return ticket.request not in self.client.active_requests

    def release(self, request):
        self.client.release(request)

    def close(self):
        # The client recovers unresolved tickets before cancellation/release and
        # refuses to reclaim outstanding native logits consumer leases.
        self.client.close()


class RetainedNativeBinding:
    @staticmethod
    def _composite_requested(options):
        enabled = options.get("composite_drain_commit", False)
        if type(enabled) is not bool:
            raise ValueError("native composite_drain_commit must be boolean")
        return enabled

    def validate_config(self, config):
        options = config.additional_config["afd_native_target"]
        composite = self._composite_requested(options)
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
            "execution_poll_interval_s",
            "prefill_mode",
            "dspark",
            "decode_batching",
            "composite_drain_commit",
        }:
            raise ValueError(
                "native retained binding needs explicit artifacts, peers "
                "and memory limits"
            )
        if options["implementation"] != "retained":
            raise ValueError("unknown native target implementation")
        execution_interval = options.get("execution_poll_interval_s")
        if execution_interval is not None and (
            type(execution_interval) not in (int, float)
            or not math.isfinite(execution_interval)
            or not 0 < execution_interval <= 1
        ):
            raise ValueError("invalid native target execution poll interval")
        batching = options.get("decode_batching", False)
        if type(batching) is not bool:
            raise ValueError("native decode_batching must be boolean")
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
        if composite:
            try:
                from vllm_afd.native_target.composite import CompositeNativeTargetClient
            except ImportError as exc:
                raise NativeTargetUnavailable(
                    "install the matching packaged composite-v1 client"
                ) from exc
            if not issubclass(CompositeNativeTargetClient, NativeTargetClient):
                raise NativeTargetUnavailable("composite client has foreign base identity")
        if not hasattr(NativeTargetClient, "can_prepare"):
            raise NativeTargetUnavailable(
                "native target client lacks authoritative capacity queries"
            )
        if dspark is not None and not hasattr(NativeTargetClient, "submit_speculative"):
            raise NativeTargetUnavailable("native target client lacks dSpark proposals")
        if batching and not all(
            hasattr(NativeTargetClient, name)
            for name in ("submit_batch", "submit_speculative_batch", "commit_batch")
        ):
            raise NativeTargetUnavailable("native target client lacks grouped decode")

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
        composite = self._composite_requested(options)
        client_type = NativeTargetClient
        if composite:
            from vllm_afd.native_target.composite import CompositeNativeTargetClient

            client_type = CompositeNativeTargetClient
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
            client_type(
                target,
                enabled=True,
                library_path=options["abi_library"],
                timeout_s=options.get("timeout_s", 60),
                poll_interval_s=options.get("poll_interval_s", 0.00005),
                execution_poll_interval_s=options.get("execution_poll_interval_s"),
                device_id=0,
            ),
            enable_composite=composite,
        )

    def create_scheduler(self, **kwargs):
        from vllm.v1.core.sched.native_scheduler import NativeScheduler

        return NativeScheduler(**kwargs)
