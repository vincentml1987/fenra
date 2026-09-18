"""Shared pytest fixtures: a tiny stdlib HTTP server standing in for a
local Ollama instance, configurable to sleep before responding (to
exercise kill/heartbeat-independence) or return malformed JSON."""
import json
import threading
import time

import pytest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOllamaConfig:
    def __init__(self):
        self.delay_seconds = 0
        self.response_body = {"response": "hi"}
        self.status_code = 200


def _make_handler(cfg):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_POST(self):
            if cfg.delay_seconds:
                time.sleep(cfg.delay_seconds)
            self.send_response(cfg.status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(cfg.response_body).encode("utf-8"))

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"models": []}).encode("utf-8"))

    return Handler


@pytest.fixture
def fake_ollama():
    cfg = FakeOllamaConfig()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(cfg))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # 127.0.0.1, not "localhost" - resolving the hostname was observed to
    # add ~2s per call on this machine, which would otherwise dominate
    # the timing-sensitive tests (kill latency, heartbeat cadence).
    cfg.host = f"http://127.0.0.1:{port}"
    yield cfg
    server.shutdown()
    server.server_close()
