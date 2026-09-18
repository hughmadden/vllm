# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disabled scheduler-side adapter for DS41RT native executor schema 1.

The native snapshot is the sole physical source-page ledger. This module owns
logical request names and command acknowledgements, never tensors or block IDs.
It cannot replace KVCacheManager until complete model checkpoints and native
weight/context accounting are connected to the worker allocation path.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from typing import Any, TypedDict


class Envelope(TypedDict):
    owner: int
    epoch: int
    command_id: int
    command: dict[str, Any]


class Acknowledgement(TypedDict):
    command_id: int
    epoch: int
    result: dict[str, Any]
    snapshot: dict[str, Any]


class AdapterError(ValueError):
    """Invalid transition, stale identity or malformed native acknowledgement."""


class CommandPending(AdapterError):
    """Retry or resolve the existing envelope before issuing another command."""


class IntegrationDisabled(AdapterError):
    """Complete model cache and physical memory integration are not available."""


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return call


def _uint(value, maximum=(1 << 64) - 1, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise AdapterError("native integer outside its declared extent")
    return value


def _lease(value, owner):
    if not isinstance(value, dict) or set(value) != {"owner", "slot", "generation"}:
        raise AdapterError("invalid native request handle")
    _uint(value["owner"], minimum=1)
    if value["owner"] != owner:
        raise AdapterError("foreign native request owner")
    _uint(value["slot"], 15)
    _uint(value["generation"], minimum=1)


def _snapshot(value):
    fields = {
        "owner",
        "epoch",
        "source_page_capacity",
        "source_pages_free",
        "cache_bytes",
        "transactions",
        "source_prefixes",
        "source_prefix_capacity",
        "requests",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise AdapterError("invalid native snapshot schema")
    _uint(value["owner"], minimum=1)
    for key in ("epoch", "cache_bytes"):
        _uint(value[key])
    _uint(value["transactions"], 2)
    _uint(value["source_prefix_capacity"], 4096)
    _uint(value["source_prefixes"], value["source_prefix_capacity"])
    for key in ("source_page_capacity", "source_pages_free"):
        if not isinstance(value[key], list) or len(value[key]) != 4:
            raise AdapterError("native source ledger requires four pools")
        for count in value[key]:
            _uint(count)
    if any(
        free > cap
        for free, cap in zip(value["source_pages_free"], value["source_page_capacity"])
    ):
        raise AdapterError("native free credits exceed physical capacity")
    requests = value["requests"]
    if not isinstance(requests, list) or len(requests) > 16:
        raise AdapterError("native request snapshot exceeds slot capacity")
    slots, names = set(), set()
    for request in requests:
        if not isinstance(request, dict) or set(request) != {
            "request",
            "request_id",
            "committed_end",
            "prepared_end",
            "readers",
            "closing",
        }:
            raise AdapterError("invalid native request snapshot")
        _lease(request["request"], value["owner"])
        _uint(request["request_id"])
        _uint(request["committed_end"], 1048576)
        _uint(request["prepared_end"], 1048576, request["committed_end"])
        _uint(request["readers"])
        if type(request["closing"]) is not bool:
            raise AdapterError("native closing state must be boolean")
        slot = request["request"]["slot"]
        if slot in slots or request["request_id"] in names:
            raise AdapterError("duplicate native slot or request ID")
        slots.add(slot)
        names.add(request["request_id"])
    return deepcopy(value)


@dataclass
class _Pending:
    envelope: Envelope
    request_id: str | None = None
    prefix_key: bytes | None = None


class NativeCacheAdapter:
    """Emit commands and consume worker ACKs on a serialized scheduler seam.

    Args:
        snapshot: Fresh, empty native incarnation snapshot (epoch zero).
        capacity_rows: Native configured row bound; no token credits are inferred.
    """

    def __init__(self, snapshot: dict[str, Any], capacity_rows: int = 4096):
        self._lock = RLock()
        self.capacity_rows = _uint(capacity_rows, 4096, 1)
        self._next_command_id = 1
        self._next_request_id = 1
        self._install(snapshot)

    def _install(self, snapshot):
        snapshot = _snapshot(snapshot)
        if (
            snapshot["epoch"]
            or snapshot["requests"]
            or snapshot["transactions"]
            or snapshot["source_prefixes"]
            or snapshot["source_pages_free"] != snapshot["source_page_capacity"]
        ):
            raise AdapterError("attachment requires a fresh empty native incarnation")
        self._snapshot = snapshot
        self._requests: dict[str, dict[str, int]] = {}
        self._transactions: dict[str, dict[str, Any]] = {}
        self._prefixes: dict[bytes, dict[str, Any]] = {}
        self._revoked: set[str] = set()
        self._generations = [0] * 16
        self._pending: _Pending | None = None
        self._last_ack: Acknowledgement | None = None

    @property
    @_locked
    def ledger(self) -> dict[str, Any]:
        """Last acknowledged native credits; cache_bytes excludes model/scratch."""
        return deepcopy(self._snapshot)

    def require_serving_ready(self) -> None:
        raise IntegrationDisabled(
            "native backend requires complete checkpoint, weight/context accounting "
            "and scheduler/worker integration before physical KV initialization"
        )

    def get_computed_blocks(self, request_id: str):
        """Source references cannot satisfy KVCacheManager's model-hit contract."""
        self.require_serving_ready()

    def _idle(self):
        if self._pending is not None:
            raise CommandPending("resolve or retry the unacknowledged native command")

    def _issue(self, op, scheduler_id=None, prefix_key=None, **fields) -> Envelope:
        self._idle()
        command_id = _uint(self._next_command_id, minimum=1)
        self._next_command_id += 1
        envelope: Envelope = {
            "owner": self._snapshot["owner"],
            "epoch": self._snapshot["epoch"],
            "command_id": command_id,
            "command": {"op": op, **deepcopy(fields)},
        }
        self._pending = _Pending(envelope, scheduler_id, prefix_key)
        return deepcopy(envelope)

    def _request(self, name, live=True):
        if name not in self._requests or (live and name in self._revoked):
            raise AdapterError("unknown or revoked scheduler request")
        return self._requests[name]

    def _state(self, name):
        handle = self._request(name, live=False)
        return next(r for r in self._snapshot["requests"] if r["request"] == handle)

    def _batch(self, name):
        self._request(name)
        if name not in self._transactions:
            raise AdapterError("request has no native transaction")
        return self._transactions[name]

    def _new_request(self, name, slot):
        self._idle()
        if not isinstance(name, str) or not 0 < len(name) <= 1024:
            raise AdapterError("invalid scheduler request name")
        _uint(slot, 15)
        if name in self._requests or any(
            h["slot"] == slot for h in self._requests.values()
        ):
            raise AdapterError("scheduler request name or native slot is occupied")
        identity = _uint(self._next_request_id, minimum=1)
        self._next_request_id += 1
        return identity

    @staticmethod
    def _key(key):
        if type(key) is not bytes or len(key) != 32:
            raise AdapterError("source prefix key must be a canonical 32-byte digest")

    @_locked
    def admit(self, request_id: str, slot: int) -> Envelope:
        identity = self._new_request(request_id, slot)
        return self._issue("admit", request_id, slot=slot, **{"request_id": identity})

    @_locked
    def begin_batch(self, request_id: str, lane: int, rows: int) -> Envelope:
        self._idle()
        handle = self._request(request_id)
        _uint(lane, 1)
        _uint(rows, self.capacity_rows, 1)
        if request_id in self._transactions or any(
            t["handle"]["lane"] == lane for t in self._transactions.values()
        ):
            raise AdapterError("request or native execution lane is busy")
        if self._state(request_id)["committed_end"] + rows > 1048576:
            raise AdapterError("proposed batch exceeds native context bound")
        return self._issue("begin", request_id, request=handle, lane=lane, rows=rows)

    @_locked
    def reserve_publish(self, request_id: str, accepted: int) -> Envelope:
        self._idle()
        batch = self._batch(request_id)
        _uint(accepted, batch["rows"])
        if batch["accepted"] is not None or batch["aborting"]:
            raise AdapterError("transaction already reserved or aborting")
        return self._issue(
            "reserve_publish",
            request_id,
            transaction=batch["handle"],
            accepted=accepted,
        )

    @_locked
    def publish(self, request_id: str) -> Envelope:
        batch = self._batch(request_id)
        if batch["accepted"] is None or batch["aborting"]:
            raise AdapterError(
                "publication requires accepted rows and a live transaction"
            )
        return self._issue("publish", request_id, transaction=batch["handle"])

    @_locked
    def abort_batch(self, request_id: str) -> Envelope:
        batch = self._batch(request_id)
        return self._issue("abort", request_id, transaction=batch["handle"])

    @_locked
    def revoke(self, request_id: str) -> None:
        """Hide local use immediately, even while an ACK is lost in transit."""
        if request_id not in self._requests and not (
            self._pending and self._pending.request_id == request_id
        ):
            raise AdapterError("unknown scheduler request")
        self._revoked.add(request_id)

    @_locked
    def free(self, request_id: str) -> Envelope:
        """Cancel/preempt logically; retain credits until native release/drain ACK."""
        self.revoke(request_id)
        self._idle()
        return self._issue(
            "release", request_id, request=self._request(request_id, live=False)
        )

    @_locked
    def drain(self, request_id: str) -> Envelope:
        if not self._state(request_id)["closing"]:
            raise AdapterError("native drain requires acknowledged release")
        return self._issue(
            "drain", request_id, request=self._request(request_id, live=False)
        )

    @_locked
    def source_committed_tokens(self, request_id: str) -> int:
        """Source extent only; never a whole-model computed-token promise."""
        self._request(request_id)
        return self._state(request_id)["committed_end"]

    @_locked
    def retain_source_prefix(self, request_id: str, key: bytes) -> Envelope:
        self._key(key)
        if key in self._prefixes or request_id in self._transactions:
            raise AdapterError(
                "prefix already retained or request transaction unfinished"
            )
        return self._issue(
            "retain_source_prefix", request_id, key, request=self._request(request_id)
        )

    @_locked
    def restore_source_prefix(self, key: bytes, request_id: str, slot: int) -> Envelope:
        self._key(key)
        if key not in self._prefixes:
            raise AdapterError("source prefix absent or reset")
        identity = self._new_request(request_id, slot)
        return self._issue(
            "restore_source_prefix",
            request_id,
            key,
            prefix=self._prefixes[key]["prefix"],
            slot=slot,
            **{"request_id": identity},
        )

    @_locked
    def lookup_source_prefix(self, key: bytes) -> dict[str, Any] | None:
        self._key(key)
        return deepcopy(self._prefixes.get(key))

    @_locked
    def drop_source_prefix(self, key: bytes) -> Envelope:
        self._key(key)
        if key not in self._prefixes:
            raise AdapterError("source prefix absent or reset")
        return self._issue(
            "drop_source_prefix", prefix_key=key, prefix=self._prefixes[key]["prefix"]
        )

    @_locked
    def reset_prefix_cache(self) -> Envelope:
        """Reset only source references; active native readers retain ownership."""
        return self._issue("reset_source_prefixes")

    @_locked
    def retry(self) -> Envelope:
        if self._pending is None:
            raise AdapterError("no native command to retry")
        return deepcopy(self._pending.envelope)

    @_locked
    def reject(self, owner: int, epoch: int, command_id: int) -> None:
        """Consume an explicit native Result::Err, never a transport timeout.

        The worker RPC must correlate this no-mutation rejection with the exact
        envelope. Ambiguous transport failures instead retain the envelope.
        """
        _uint(owner, minimum=1)
        _uint(epoch)
        _uint(command_id, minimum=1)
        if self._pending is None or any(
            self._pending.envelope[k] != v
            for k, v in (("owner", owner), ("epoch", epoch), ("command_id", command_id))
        ):
            raise AdapterError("rejection does not match the pending native command")
        if self._pending.request_id not in self._requests:
            self._revoked.discard(self._pending.request_id)
        self._pending = None

    @_locked
    def restart(self, snapshot: dict[str, Any]) -> tuple[str, ...]:
        """Invalidate local state after a confirmed worker exit/new incarnation.

        Caller supplies a never-reused native owner nonce. Old storage cleanup
        is owned by the dead worker's lifecycle, not inferred from new credits.
        Requests must recompute; source references are insufficient for resume.
        """
        snapshot = _snapshot(snapshot)
        if snapshot["owner"] == self._snapshot["owner"]:
            raise AdapterError("worker restart requires a new native owner")
        invalidated = set(self._requests)
        if self._pending and self._pending.request_id:
            invalidated.add(self._pending.request_id)
        self._install(snapshot)
        return tuple(sorted(invalidated))

    @_locked
    def acknowledge(self, acknowledgement: Acknowledgement) -> bool:
        """Publish a validated ACK atomically; malformed ACKs leave retry intact."""
        if self._last_ack is not None and acknowledgement == self._last_ack:
            return False
        if self._pending is None:
            raise AdapterError("unsolicited native acknowledgement")
        if not isinstance(acknowledgement, dict) or set(acknowledgement) != {
            "command_id",
            "epoch",
            "result",
            "snapshot",
        }:
            raise AdapterError("invalid native acknowledgement schema")
        pending = self._pending
        envelope = pending.envelope
        snapshot = _snapshot(acknowledgement["snapshot"])
        _uint(acknowledgement["command_id"], minimum=1)
        _uint(acknowledgement["epoch"], minimum=1)
        if (
            acknowledgement["command_id"] != envelope["command_id"]
            or acknowledgement["epoch"] != envelope["epoch"] + 1
            or snapshot["epoch"] != acknowledgement["epoch"]
            or snapshot["owner"] != envelope["owner"]
        ):
            raise AdapterError("stale or foreign native acknowledgement")
        for key in ("source_page_capacity", "cache_bytes", "source_prefix_capacity"):
            if snapshot[key] != self._snapshot[key]:
                raise AdapterError(
                    "physical native capacity changed within an incarnation"
                )
        command = envelope["command"]
        op, name, key = command["op"], pending.request_id, pending.prefix_key
        result = deepcopy(acknowledgement["result"])
        allowed = {
            "admit": {"admitted"},
            "restore_source_prefix": {"admitted"},
            "begin": {"begun"},
            "reserve_publish": {"reserved", "failed"},
            "publish": {"pending", "published", "failed"},
            "abort": {"pending", "aborted"},
            "release": {"closing", "released"},
            "drain": {"closing", "released"},
            "retain_source_prefix": {"source_prefix"},
            "drop_source_prefix": {"prefixes_dropped"},
            "reset_source_prefixes": {"prefixes_dropped"},
        }
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("state"), str)
            or result["state"] not in allowed[op]
        ):
            raise AdapterError("native outcome does not match the pending command")
        state = result["state"]
        extra_fields = {
            "admitted": {"request"},
            "begun": {"transaction"},
            "published": {"committed_end"},
            "source_prefix": {"prefix", "committed_end"},
            "failed": {"reason", "draining"},
        }
        if set(result) != {"state"} | extra_fields.get(state, set()):
            raise AdapterError("invalid native outcome fields")
        requests, transactions, prefixes = deepcopy(
            (self._requests, self._transactions, self._prefixes)
        )
        ends = {n: self._state(n)["committed_end"] for n in requests}
        native_ids = {n: self._state(n)["request_id"] for n in requests}
        closing = {n: self._state(n)["closing"] for n in requests}
        generations = self._generations.copy()
        try:
            if state == "admitted":
                h = result["request"]
                _lease(h, snapshot["owner"])
                if (
                    h["slot"] != command["slot"]
                    or h["generation"] <= generations[h["slot"]]
                ):
                    raise AdapterError("native admission reused a stale generation")
                requests[name] = h
                generations[h["slot"]] = h["generation"]
                ends[name] = 0 if op == "admit" else prefixes[key]["committed_end"]
                native_ids[name] = command["request_id"]
                closing[name] = False
            elif state == "begun":
                t = result["transaction"]
                if not isinstance(t, dict) or set(t) != {
                    "owner",
                    "id",
                    "lane",
                    "request",
                }:
                    raise AdapterError("invalid native transaction")
                _uint(t["id"], minimum=1)
                _uint(t["lane"], 1)
                _uint(t["owner"], minimum=1)
                _lease(t["request"], snapshot["owner"])
                if (
                    t["owner"] != snapshot["owner"]
                    or t["request"] != requests[name]
                    or t["lane"] != command["lane"]
                ):
                    raise AdapterError(
                        "native transaction does not own the requested lane"
                    )
                transactions[name] = {
                    "handle": t,
                    "rows": command["rows"],
                    "accepted": None,
                    "aborting": False,
                }
            elif state == "reserved":
                transactions[name]["accepted"] = command["accepted"]
            elif state == "failed":
                if (
                    type(result["draining"]) is not bool
                    or result["draining"] != (op == "reserve_publish")
                    or not isinstance(result["reason"], str)
                    or len(result["reason"]) > 4096
                ):
                    raise AdapterError("invalid native write-failure state")
                if result["draining"]:
                    transactions[name]["aborting"] = True
                else:
                    transactions.pop(name)
            elif state == "published":
                _uint(result["committed_end"], 1048576)
                accepted = transactions[name]["accepted"]
                if accepted is None or result["committed_end"] != ends[name] + accepted:
                    raise AdapterError("native publication exceeds accepted extent")
                ends[name] = result["committed_end"]
                transactions.pop(name)
            elif state == "aborted":
                transactions.pop(name)
            elif state == "released":
                requests.pop(name)
                transactions.pop(name, None)
            elif state == "closing":
                current = next(
                    r for r in snapshot["requests"] if r["request"] == requests[name]
                )
                if not current["closing"]:
                    raise AdapterError("native release did not revoke the request")
                closing[name] = True
                if current["prepared_end"] == current["committed_end"]:
                    transactions.pop(name, None)
                elif name in transactions:
                    transactions[name]["aborting"] = True
            elif state == "source_prefix":
                p = result["prefix"]
                if (
                    not isinstance(p, dict)
                    or set(p) != {"owner", "id"}
                    or p["owner"] != snapshot["owner"]
                    or result["committed_end"] != ends[name]
                ):
                    raise AdapterError("invalid native source-prefix acknowledgement")
                _uint(p["id"], minimum=1)
                _uint(p["owner"], minimum=1)
                _uint(result["committed_end"], 1048576)
                if any(v["prefix"] == p for v in prefixes.values()):
                    raise AdapterError("native prefix identity already retained")
                prefixes[key] = {"prefix": p, "committed_end": ends[name]}
            elif state == "prefixes_dropped":
                if op == "reset_source_prefixes":
                    prefixes.clear()
                else:
                    prefixes.pop(key)
            if state == "pending" and op == "abort":
                transactions[name]["aborting"] = True
            if (
                len(snapshot["requests"]) != len(requests)
                or snapshot["transactions"] != len(transactions)
                or snapshot["source_prefixes"] != len(prefixes)
            ):
                raise AdapterError(
                    "native snapshot ownership counts disagree with acknowledgement"
                )
            for n, h in requests.items():
                r = next(r for r in snapshot["requests"] if r["request"] == h)
                proposed = transactions[n]["rows"] if n in transactions else 0
                if (
                    r["request_id"] != native_ids[n]
                    or r["committed_end"] != ends[n]
                    or r["prepared_end"] != ends[n] + proposed
                    or r["closing"] != closing[n]
                ):
                    raise AdapterError("unacknowledged native request extent changed")
        except (KeyError, TypeError, StopIteration) as exc:
            raise AdapterError("incomplete native acknowledgement") from exc
        self._snapshot = snapshot
        self._requests, self._transactions, self._prefixes = (
            requests,
            transactions,
            prefixes,
        )
        self._generations = generations
        if state == "released":
            self._revoked.discard(name)
        self._last_ack = deepcopy(acknowledgement)
        self._pending = None
        return True
