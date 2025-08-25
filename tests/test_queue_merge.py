import asyncio
from conductor import merge_and_clear_queue, inject_external_message


def test_merge_and_clear_queue():
    asyncio.run(inject_external_message("hi", {"author": "b", "timestamp": "2024-01-02"}))
    asyncio.run(inject_external_message("hey", {"author": "a", "timestamp": "2024-01-01"}))
    text = merge_and_clear_queue()
    assert text.splitlines()[0].startswith("[2024-01-01] a:")
    assert merge_and_clear_queue() == ""
