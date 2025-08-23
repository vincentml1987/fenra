import json
from conductor import merge_and_clear_queue


def test_merge_and_clear_queue(tmp_path):
    path = tmp_path / "queue.json"
    items = [
        {"timestamp": "2024-01-02", "sender": "b", "message": "hi"},
        {"timestamp": "2024-01-01", "sender": "a", "message": "hey"},
    ]
    path.write_text(json.dumps(items), encoding="utf-8")
    text = merge_and_clear_queue(path)
    assert text.splitlines()[0].startswith("[2024-01-01] a:")
    assert json.loads(path.read_text(encoding="utf-8")) == []
