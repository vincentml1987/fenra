from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json, os, time

from config_loader import get_path

DEFAULT_DM_PATH = Path("confs") / "directed_memories.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp-{int(time.time() * 1000)}")
    os.makedirs(path.parent, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def directed_memories_path() -> Path:
    p = get_path("paths.directed_memories", default=DEFAULT_DM_PATH)
    if p is None:
        p = DEFAULT_DM_PATH
    return Path(p)


@dataclass
class DirectedMemory:
    id: int
    isGlobal: bool
    agentClasses: List[str]
    agents: List[str]
    memoryText: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DirectedMemory":
        return DirectedMemory(
            id=int(d.get("id")),
            isGlobal=bool(d.get("isGlobal", False)),
            agentClasses=[str(x) for x in (d.get("agentClasses") or [])],
            agents=[str(x) for x in (d.get("agents") or [])],
            memoryText=str(d.get("memoryText", "")),
        )


class DirectedMemoryStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or directed_memories_path())

    def _ensure_file(self) -> None:
        if not self.path.exists():
            _atomic_write_json(self.path, {"memories": []})

    def _load_payload(self) -> Dict[str, Any]:
        self._ensure_file()
        try:
            with self.path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            payload = {"memories": []}
        if not isinstance(payload, dict):
            payload = {"memories": []}
        payload.setdefault("memories", [])
        return payload

    def _save_payload(self, payload: Dict[str, Any]) -> None:
        _atomic_write_json(self.path, payload)

    def list(self) -> List[DirectedMemory]:
        payload = self._load_payload()
        out: List[DirectedMemory] = []
        for item in payload.get("memories", []):
            try:
                out.append(DirectedMemory.from_dict(item))
            except Exception:
                continue
        out.sort(key=lambda m: m.id)
        return out

    def list_raw(self) -> List[Dict[str, Any]]:
        return [asdict(m) for m in self.list()]

    def _next_id(self) -> int:
        items = self.list()
        return (max((m.id for m in items), default=0) + 1)

    def add(self, memory_text: str) -> DirectedMemory:
        mem = DirectedMemory(
            id=self._next_id(),
            isGlobal=True,
            agentClasses=[],
            agents=[],
            memoryText=str(memory_text or "").strip(),
        )
        payload = self._load_payload()
        payload.setdefault("memories", []).append(asdict(mem))
        self._save_payload(payload)
        return mem

    def _find_index(self, idx: int) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        payload = self._load_payload()
        arr = payload.setdefault("memories", [])
        for i, item in enumerate(arr):
            if int(item.get("id", -1)) == int(idx):
                return i, item, payload
        raise KeyError(f"directed memory {idx} not found")

    def delete(self, idx: int) -> None:
        i, _item, payload = self._find_index(idx)
        del payload["memories"][i]
        self._save_payload(payload)

    def update_text(self, idx: int, memory_text: str) -> Dict[str, Any]:
        i, item, payload = self._find_index(idx)
        item["memoryText"] = str(memory_text or "").strip()
        payload["memories"][i] = item
        self._save_payload(payload)
        return item

    def set_global(self, idx: int, is_global: bool) -> Dict[str, Any]:
        i, item, payload = self._find_index(idx)
        item["isGlobal"] = bool(is_global)
        payload["memories"][i] = item
        self._save_payload(payload)
        return item

    def add_agent_class(self, idx: int, agent_class: str) -> Dict[str, Any]:
        i, item, payload = self._find_index(idx)
        classes = [str(x) for x in (item.get("agentClasses") or [])]
        val = str(agent_class or "").strip()
        if val and val not in classes:
            classes.append(val)
        item["agentClasses"] = classes
        payload["memories"][i] = item
        self._save_payload(payload)
        return item

    def remove_agent_class(self, idx: int, agent_class: str) -> Dict[str, Any]:
        i, item, payload = self._find_index(idx)
        val = str(agent_class or "").strip()
        classes = [c for c in (item.get("agentClasses") or []) if str(c) != val]
        item["agentClasses"] = classes
        payload["memories"][i] = item
        self._save_payload(payload)
        return item

    def add_agent(self, idx: int, agent: str) -> Dict[str, Any]:
        i, item, payload = self._find_index(idx)
        agents = [str(x) for x in (item.get("agents") or [])]
        val = str(agent or "").strip()
        if val and val not in agents:
            agents.append(val)
        item["agents"] = agents
        payload["memories"][i] = item
        self._save_payload(payload)
        return item

    def remove_agent(self, idx: int, agent: str) -> Dict[str, Any]:
        i, item, payload = self._find_index(idx)
        val = str(agent or "").strip()
        agents = [a for a in (item.get("agents") or []) if str(a) != val]
        item["agents"] = agents
        payload["memories"][i] = item
        self._save_payload(payload)
        return item

    def select_for(self, *, agent_name: str, agent_class: str) -> List[str]:
        out: List[str] = []
        for m in self.list():
            if m.isGlobal or (agent_class in m.agentClasses) or (agent_name in m.agents):
                if m.memoryText.strip():
                    out.append(m.memoryText.strip())
        return out


def get_store() -> DirectedMemoryStore:
    return DirectedMemoryStore()


def format_directed_memories_block(lines: Iterable[str]) -> str:
    items = [str(x).strip() for x in (lines or []) if str(x).strip()]
    if not items:
        return ""
    block = ["**********Directed Memories Begin**********"]
    block.extend(items)
    block.append("**********Directed Memories End**********")
    return "\n".join(block)


def directed_memory_block_for_agent(agent_name: str, agent_class: str) -> str:
    try:
        store = get_store()
        lines = store.select_for(agent_name=agent_name, agent_class=agent_class)
        return format_directed_memories_block(lines)
    except Exception:
        return ""
