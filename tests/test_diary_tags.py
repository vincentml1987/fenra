import json
from pathlib import Path

import pytest

from diary_tags import (
    DiaryPaths,
    add_file_tags,
    get_file_tags,
    list_tree,
    load_index,
    mkdir,
    move,
    reindex,
    remove,
    remove_file_tags,
    search_by_tags,
    set_file_tags,
)


def _make_paths(tmp_path: Path, *, doc_inside_diary: bool = False) -> DiaryPaths:
    diary_root = tmp_path / "diary"
    documentation_root = tmp_path / "documentation"
    diary_root.mkdir(parents=True, exist_ok=True)
    documentation_root.mkdir(parents=True, exist_ok=True)
    if doc_inside_diary:
        inner_doc = diary_root / "documentation"
        inner_doc.mkdir(parents=True, exist_ok=True)
        documentation_root = inner_doc
    return DiaryPaths(diary_root=diary_root, documentation_root=documentation_root)


def _write_file(paths: DiaryPaths, relative: str, content: str = "entry") -> Path:
    file_path = paths.diary_root / relative
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_path_escape_and_documentation_block(tmp_path):
    paths = _make_paths(tmp_path)

    with pytest.raises(PermissionError):
        mkdir(paths, "../outside")

    with pytest.raises(PermissionError):
        list_tree(paths, "/absolute")

    inner_paths = _make_paths(tmp_path / "nested", doc_inside_diary=True)
    with pytest.raises(PermissionError):
        mkdir(inner_paths, "documentation/new")


def test_directory_operations(tmp_path):
    paths = _make_paths(tmp_path)
    mkdir(paths, "project")
    _write_file(paths, "note.txt", "hello")

    tree = list_tree(paths, ".")
    assert tree["dirs"] == ["project"]
    assert tree["files"] == ["note.txt"]

    add_file_tags(paths, "note.txt", ["Focus"])  # create sidecar to move
    moved = move(paths, "note.txt", "project/journal.txt")
    assert moved == "project/journal.txt"
    assert (paths.diary_root / "project" / "journal.txt").exists()
    assert not (paths.diary_root / "note.txt").exists()
    assert load_index(paths) == {"project/journal.txt": ["Focus"]}

    removed = remove(paths, "project")
    assert removed is True
    assert not (paths.diary_root / "project").exists()
    assert load_index(paths) == {}

    assert remove(paths, "missing.txt") is False


def test_tag_lifecycle_and_search(tmp_path):
    paths = _make_paths(tmp_path)
    _write_file(paths, "entry.md", "dear diary")

    assert get_file_tags(paths, "entry.md") == []

    tags = add_file_tags(paths, "entry.md", ["Focus", "daily"])
    assert tags == ["daily", "Focus"]

    tags = add_file_tags(paths, "entry.md", ["focus", "Plan"])
    assert tags == ["daily", "Focus", "Plan"]

    tags = remove_file_tags(paths, "entry.md", ["FOCUS"])
    assert tags == ["daily", "Plan"]

    tags = set_file_tags(paths, "entry.md", ["Alpha", "beta"])
    assert tags == ["Alpha", "beta"]

    _write_file(paths, "other.txt", "notes")
    add_file_tags(paths, "other.txt", ["beta", "gamma"])

    matches = search_by_tags(paths, include=["beta"])
    assert matches == ["entry.md", "other.txt"]

    matches = search_by_tags(paths, include=["beta"], exclude=["gamma"])
    assert matches == ["entry.md"]


def test_sidecar_and_index_persistence(tmp_path):
    paths = _make_paths(tmp_path)
    _write_file(paths, "persist.txt", "keep")
    add_file_tags(paths, "persist.txt", ["Keep"])

    # Simulate new process
    paths2 = DiaryPaths(paths.diary_root, paths.documentation_root)
    assert get_file_tags(paths2, "persist.txt") == ["Keep"]

    index_path = paths.diary_root / ".fenra" / "tags_index.json"
    assert index_path.exists()
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_data["files"] == {"persist.txt": ["Keep"]}

    index_path.unlink()
    count = reindex(paths2)
    assert count == 1
    rebuilt = json.loads(index_path.read_text(encoding="utf-8"))
    assert rebuilt["files"] == {"persist.txt": ["Keep"]}
