"""Diary directory management and tagging restricted to the diary root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json
import os
import shutil
import time


@dataclass(frozen=True)
class DiaryPaths:
    """Resolved paths used for diary operations."""

    diary_root: Path
    documentation_root: Optional[Path] = None


# -------- Path guards --------
def _resolve_in_diary(paths: DiaryPaths, rel: str | Path) -> Path:
    """Resolve a diary-relative path and ensure it never escapes the diary root."""

    rel_path = Path(rel)
    if rel_path.is_absolute():
        raise PermissionError("Absolute paths are not allowed.")

    diary_root = paths.diary_root.resolve()
    target = (paths.diary_root / rel_path).resolve()

    if paths.documentation_root is not None:
        doc_root = paths.documentation_root.resolve()
        try:
            target.relative_to(doc_root)
        except ValueError:
            pass
        else:
            raise PermissionError("Writes to documentation are not allowed.")

    try:
        target.relative_to(diary_root)
    except ValueError as exc:
        raise PermissionError("Path escapes the diary root.") from exc

    return target


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(target: Path, payload: dict) -> None:
    tmp = target.with_suffix(target.suffix + f".tmp-{int(time.time() * 1000)}")
    _ensure_dir(tmp.parent)
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, target)


# -------- Tag sidecar helpers --------
def sidecar_path(file_path: Path) -> Path:
    return file_path.with_suffix(file_path.suffix + ".tags.json")


def _normalize_tags(tags: Iterable[str]) -> List[str]:
    seen: Dict[str, str] = {}
    for tag in tags:
        if not isinstance(tag, str):
            continue
        cleaned = tag.strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        if lower not in seen:
            seen[lower] = cleaned
    return sorted(seen.values(), key=str.lower)


def read_tags(file_path: Path) -> List[str]:
    sc_path = sidecar_path(file_path)
    if not sc_path.exists():
        return []
    try:
        data = json.loads(sc_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    tags = data.get("tags", [])
    return _normalize_tags(tags)


def write_tags(file_path: Path, tags: Iterable[str]) -> List[str]:
    ordered = _normalize_tags(tags)
    sidecar = sidecar_path(file_path)
    _atomic_write_json(sidecar, {"tags": ordered})
    return ordered


# -------- Index helpers --------
def _index_file_map(paths: DiaryPaths) -> Path:
    fenra_dir = paths.diary_root / ".fenra"
    _ensure_dir(fenra_dir)
    return fenra_dir / "tags_index.json"


def load_index(paths: DiaryPaths) -> Dict[str, List[str]]:
    index_path = _index_file_map(paths)
    if not index_path.exists():
        return {}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    files = data.get("files", {})
    normalized: Dict[str, List[str]] = {}
    if isinstance(files, dict):
        for rel, tags in files.items():
            if not isinstance(rel, str):
                continue
            normalized_tags = _normalize_tags(tags if isinstance(tags, list) else [])
            if normalized_tags:
                normalized[rel] = normalized_tags
    return normalized


def save_index(paths: DiaryPaths, files_map: Dict[str, List[str]]) -> None:
    payload_files: Dict[str, List[str]] = {}
    for rel, tags in files_map.items():
        normalized_tags = _normalize_tags(tags)
        if normalized_tags:
            payload_files[rel] = normalized_tags

    payload = {
        "version": 1,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": payload_files,
    }
    _atomic_write_json(_index_file_map(paths), payload)


def reindex(paths: DiaryPaths) -> int:
    files_map: Dict[str, List[str]] = {}
    diary_root = paths.diary_root.resolve()
    for item in diary_root.rglob("*"):
        if not item.is_file():
            continue
        try:
            rel_path = item.relative_to(diary_root)
        except ValueError:
            continue
        if rel_path.parts and rel_path.parts[0] == ".fenra":
            continue
        if item.name.endswith(".tags.json"):
            continue
        tags = read_tags(item)
        if tags:
            files_map[rel_path.as_posix()] = tags
    save_index(paths, files_map)
    return len(files_map)


# -------- Directory operations --------
def list_tree(paths: DiaryPaths, rel_dir: str = ".") -> Dict[str, object]:
    base = _resolve_in_diary(paths, rel_dir)
    rel_display = Path(rel_dir)
    result: Dict[str, object] = {"path": str(rel_display), "dirs": [], "files": []}

    if not base.exists():
        return result  # type: ignore[return-value]

    dirs: List[str] = []
    files: List[str] = []
    for child in sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if child.name.endswith(".tags.json"):
            continue
        if child.is_dir():
            if child.name == ".fenra":
                continue
            dirs.append(child.name)
        elif child.is_file():
            files.append(child.name)

    result["dirs"] = dirs
    result["files"] = files
    return result


def mkdir(paths: DiaryPaths, rel_dir: str) -> str:
    directory = _resolve_in_diary(paths, rel_dir)
    _ensure_dir(directory)
    return directory.relative_to(paths.diary_root.resolve()).as_posix()


def move(paths: DiaryPaths, rel_src: str, rel_dst: str, overwrite: bool = False) -> str:
    src = _resolve_in_diary(paths, rel_src)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {rel_src}")

    dst = _resolve_in_diary(paths, rel_dst)

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {rel_dst}")
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    _ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))

    if dst.is_file():
        src_sidecar = sidecar_path(src)
        if src_sidecar.exists():
            dst_sidecar = sidecar_path(dst)
            _ensure_dir(dst_sidecar.parent)
            shutil.move(str(src_sidecar), str(dst_sidecar))

    reindex(paths)
    return dst.relative_to(paths.diary_root.resolve()).as_posix()


def remove(paths: DiaryPaths, rel_path: str) -> bool:
    target = _resolve_in_diary(paths, rel_path)
    if not target.exists():
        return False

    if target.is_dir():
        for sidecar in target.rglob("*.tags.json"):
            try:
                sidecar.unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(target)
    else:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        sidecar = sidecar_path(target)
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass

    reindex(paths)
    return True


# -------- Tag operations --------
def get_file_tags(paths: DiaryPaths, rel_file: str) -> List[str]:
    file_path = _resolve_in_diary(paths, rel_file)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(rel_file)
    return read_tags(file_path)


def add_file_tags(paths: DiaryPaths, rel_file: str, tags: Iterable[str]) -> List[str]:
    file_path = _resolve_in_diary(paths, rel_file)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(rel_file)

    existing = read_tags(file_path)
    combined = existing + [t for t in tags if isinstance(t, str)]
    final = write_tags(file_path, combined)

    index = load_index(paths)
    rel_key = file_path.relative_to(paths.diary_root.resolve()).as_posix()
    if final:
        index[rel_key] = final
    else:
        index.pop(rel_key, None)
    save_index(paths, index)
    return final


def remove_file_tags(paths: DiaryPaths, rel_file: str, tags: Iterable[str]) -> List[str]:
    file_path = _resolve_in_diary(paths, rel_file)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(rel_file)

    existing = read_tags(file_path)
    removals = {t.strip().lower() for t in tags if isinstance(t, str) and t.strip()}
    remaining = [tag for tag in existing if tag.lower() not in removals]
    final = write_tags(file_path, remaining)

    index = load_index(paths)
    rel_key = file_path.relative_to(paths.diary_root.resolve()).as_posix()
    if final:
        index[rel_key] = final
    else:
        index.pop(rel_key, None)
    save_index(paths, index)
    return final


def set_file_tags(paths: DiaryPaths, rel_file: str, tags: Iterable[str]) -> List[str]:
    file_path = _resolve_in_diary(paths, rel_file)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(rel_file)

    final = write_tags(file_path, tags)
    index = load_index(paths)
    rel_key = file_path.relative_to(paths.diary_root.resolve()).as_posix()
    if final:
        index[rel_key] = final
    else:
        index.pop(rel_key, None)
    save_index(paths, index)
    return final


def search_by_tags(paths: DiaryPaths, include: Iterable[str], exclude: Iterable[str] = ()) -> List[str]:
    include_set = {t.strip().lower() for t in include if isinstance(t, str) and t.strip()}
    exclude_set = {t.strip().lower() for t in exclude if isinstance(t, str) and t.strip()}

    index = load_index(paths)
    matches: List[str] = []
    for rel, tags in index.items():
        lower_tags = {t.lower() for t in tags}
        if include_set and not include_set.issubset(lower_tags):
            continue
        if exclude_set and lower_tags & exclude_set:
            continue
        matches.append(rel)

    matches.sort()
    return matches

