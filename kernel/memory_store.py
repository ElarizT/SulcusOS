from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_DIR = Path(os.getenv("AGENT_OS_MEMORY_DIR", ".agent_os/memory"))


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _safe_json(value: Any) -> Any:
    json.dumps(value)
    return value


@dataclass
class MemoryRecord:
    memory_id: str
    owner_pid: int | None
    process_name: str
    timestamp: float
    content: Any
    token_estimate: int
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)
    tier: str = "hot"
    last_accessed: float = field(default_factory=_now)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=str(data["memory_id"]),
            owner_pid=None if data.get("owner_pid") is None else int(data["owner_pid"]),
            process_name=str(data["process_name"]),
            timestamp=float(data["timestamp"]),
            content=data.get("content"),
            token_estimate=max(int(data.get("token_estimate", 1)), 1),
            importance=float(data.get("importance", 0.5)),
            tags=[str(tag) for tag in data.get("tags", [])],
            source=dict(data.get("source", {})),
            tier=str(data.get("tier", "cold")),
            last_accessed=float(data.get("last_accessed", data.get("timestamp", _now()))),
        )


@dataclass
class AgentMemoryState:
    process_name: str
    token_budget: int
    owner_pid: int | None = None
    hot: list[MemoryRecord] = field(default_factory=list)
    warm: list[MemoryRecord] = field(default_factory=list)
    last_eviction_time: float | None = None

    @property
    def hot_tokens(self) -> int:
        return sum(record.token_estimate for record in self.hot)


class PersistentMemoryManager:
    """Tiered hot/warm/cold memory manager with JSONL persistence."""

    def __init__(self, memory_dir: Path | str | None = None, warm_limit: int = 128) -> None:
        self.memory_dir = Path(memory_dir or os.getenv("AGENT_OS_MEMORY_DIR", str(DEFAULT_MEMORY_DIR))).expanduser().resolve()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.memory_dir / "memories.jsonl"
        self.snapshots_dir = self.memory_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.warm_limit = max(int(warm_limit), 1)
        self._agents: dict[str, AgentMemoryState] = {}
        self._cold_index: dict[str, MemoryRecord] = {}
        self._load_cold_index()

    def register_agent(self, agent_name: str, max_active_tokens: int) -> None:
        if not agent_name:
            raise ValueError("agent_name must not be empty")
        if max_active_tokens <= 0:
            raise ValueError("max_active_tokens must be greater than zero")
        existing = self._agents.get(agent_name)
        owner_pid = existing.owner_pid if existing else None
        self._agents[agent_name] = AgentMemoryState(agent_name, int(max_active_tokens), owner_pid=owner_pid)

    def bind_process(self, agent_name: str, owner_pid: int) -> None:
        self._state(agent_name).owner_pid = int(owner_pid)

    def unregister_agent(self, agent_name: str) -> bool:
        return self._agents.pop(agent_name, None) is not None

    def append_context_frame(
        self,
        agent_name: str,
        content: Any,
        token_estimate: int,
        *,
        importance: float = 0.5,
        tags: list[str] | tuple[str, ...] | None = None,
        source: dict[str, Any] | None = None,
    ) -> bool:
        state = self._state(agent_name)
        record = MemoryRecord(
            memory_id=_new_id("mem"),
            owner_pid=state.owner_pid,
            process_name=agent_name,
            timestamp=_now(),
            content=_safe_json(content),
            token_estimate=max(int(token_estimate), 1),
            importance=max(0.0, min(float(importance), 1.0)),
            tags=[str(tag) for tag in (tags or [])],
            source=dict(source or {}),
            tier="hot",
        )
        state.hot.append(record)
        return self._evict_if_needed(state)

    def get_active_context(self, agent_name: str) -> list[str]:
        return [_json_text(record.content) for record in self._state(agent_name).hot]

    def get_page_table_summary(self, agent_name: str) -> dict[str, Any]:
        state = self._state(agent_name)
        warm_ids = {record.memory_id for record in state.warm}
        cold_count = sum(1 for record in self._cold_index.values() if record.process_name == agent_name)
        cold_unique_count = sum(
            1
            for record in self._cold_index.values()
            if record.process_name == agent_name and record.memory_id not in warm_ids
        )
        return {
            "agent_name": agent_name,
            "current_active_tokens": state.hot_tokens,
            "max_active_tokens": state.token_budget,
            "active_frames": len(state.hot),
            "warm_frames": len(state.warm),
            "paged_out_frames": len(state.warm) + cold_unique_count,
            "total_frames": len(state.hot) + len(state.warm) + cold_unique_count,
            "pending_evictions": 0,
            "cold_frames": cold_count,
            "snapshot_count": self.snapshot_count(agent_name),
            "last_eviction_time": state.last_eviction_time,
            "memory_store_size_bytes": self.store_size_bytes(),
        }

    def list_agents(self) -> list[str]:
        return sorted(self._agents)

    def get_global_active_token_count(self) -> int:
        return sum(state.hot_tokens for state in self._agents.values())

    def recall(
        self,
        agent_name: str,
        *,
        query: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        tag_set = {str(tag).lower() for tag in (tags or [])}
        query_text = (query or "").lower()
        candidates_by_id: dict[str, MemoryRecord] = {}
        for state in self._agents.values():
            for record in [*state.hot, *state.warm]:
                candidates_by_id.setdefault(record.memory_id, record)
        for record in self._cold_index.values():
            candidates_by_id.setdefault(record.memory_id, record)
        candidates = list(candidates_by_id.values())

        def matches(record: MemoryRecord) -> bool:
            if tag_set and not tag_set.intersection(tag.lower() for tag in record.tags):
                return False
            if query_text and query_text not in _json_text(record.content).lower():
                return False
            return True

        found = [record for record in candidates if matches(record)]
        found.sort(key=lambda item: (item.importance, item.timestamp), reverse=True)
        now = _now()
        for record in found[:limit]:
            record.last_accessed = now
        return [asdict(record) for record in found[: max(int(limit), 0)]]

    def forget(self, memory_id: str) -> bool:
        removed = False
        for state in self._agents.values():
            before_hot = len(state.hot)
            before_warm = len(state.warm)
            state.hot = [record for record in state.hot if record.memory_id != memory_id]
            state.warm = [record for record in state.warm if record.memory_id != memory_id]
            removed = removed or len(state.hot) != before_hot or len(state.warm) != before_warm
        if memory_id in self._cold_index:
            self._cold_index.pop(memory_id, None)
            removed = True
            self._rewrite_records_file()
        return removed

    def snapshot_process(self, pid: int, process_name: str) -> str:
        state = self._state(process_name)
        snapshot_id = _new_id("snap")
        payload = {
            "snapshot_id": snapshot_id,
            "pid": int(pid),
            "process_name": process_name,
            "timestamp": _now(),
            "hot": [asdict(record) for record in state.hot],
            "warm_refs": [record.memory_id for record in state.warm],
            "cold_refs": [
                record.memory_id for record in self._cold_index.values() if record.process_name == process_name
            ],
            "token_usage": state.hot_tokens,
            "token_budget": state.token_budget,
        }
        path = self.snapshots_dir / f"{snapshot_id}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return snapshot_id

    def restore_process_memory(
        self,
        pid: int,
        process_name: str,
        snapshot_id: str | None = None,
        *,
        hot_only: bool = False,
    ) -> str | None:
        snapshot = self._load_snapshot(process_name, snapshot_id)
        if snapshot is None:
            return None
        state = self._state(process_name)
        state.owner_pid = int(pid)
        state.hot = []
        for raw in snapshot.get("hot", []):
            record = MemoryRecord.from_dict(raw)
            record.owner_pid = int(pid)
            record.process_name = process_name
            record.tier = "hot"
            state.hot.append(record)
        if not hot_only:
            state.warm = [
                record
                for memory_id in snapshot.get("warm_refs", [])
                if (record := self._cold_index.get(str(memory_id))) is not None
            ][: self.warm_limit]
        self._evict_if_needed(state)
        return str(snapshot["snapshot_id"])

    def latest_snapshot_id(self, process_name: str) -> str | None:
        snapshots = self._snapshots_for(process_name)
        if not snapshots:
            return None
        snapshots.sort(key=lambda item: float(item.get("timestamp", 0.0)), reverse=True)
        return str(snapshots[0]["snapshot_id"])

    def snapshot_count(self, process_name: str) -> int:
        return len(self._snapshots_for(process_name))

    def store_size_bytes(self) -> int:
        total = 0
        for path in [self.records_path, *self.snapshots_dir.glob("*.json")]:
            if path.exists():
                total += path.stat().st_size
        return total

    def _evict_if_needed(self, state: AgentMemoryState) -> bool:
        evicted = False
        target_tokens = max(int(state.token_budget * 0.7), 1)
        while state.hot_tokens > state.token_budget and state.hot:
            candidate = min(state.hot, key=lambda item: (item.importance, item.last_accessed, item.timestamp))
            state.hot.remove(candidate)
            candidate.tier = "warm"
            state.warm.append(candidate)
            self._persist_record(candidate)
            state.last_eviction_time = _now()
            evicted = True
            if len(state.warm) > self.warm_limit:
                state.warm = state.warm[-self.warm_limit :]
            if state.hot_tokens <= target_tokens:
                break
        return evicted

    def _persist_record(self, record: MemoryRecord) -> None:
        cold = MemoryRecord.from_dict(asdict(record))
        cold.tier = "cold"
        self._cold_index[cold.memory_id] = cold
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(cold.to_json() + "\n")

    def _load_cold_index(self) -> None:
        if not self.records_path.exists():
            return
        for line in self.records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = MemoryRecord.from_dict(json.loads(line))
            record.tier = "cold"
            self._cold_index[record.memory_id] = record

    def _rewrite_records_file(self) -> None:
        with self.records_path.open("w", encoding="utf-8") as handle:
            for record in self._cold_index.values():
                handle.write(record.to_json() + "\n")

    def _load_snapshot(self, process_name: str, snapshot_id: str | None) -> dict[str, Any] | None:
        if snapshot_id is None:
            snapshot_id = self.latest_snapshot_id(process_name)
        if snapshot_id is None:
            return None
        path = self.snapshots_dir / f"{snapshot_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _snapshots_for(self, process_name: str) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        for path in self.snapshots_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("process_name") == process_name:
                snapshots.append(payload)
        return snapshots

    def _state(self, agent_name: str) -> AgentMemoryState:
        if agent_name not in self._agents:
            self.register_agent(agent_name, 8000)
        return self._agents[agent_name]
