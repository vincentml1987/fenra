# Stretch benchmark, partial: num_ctx 12288, num_predict 3000, thinking on

Qualia, Architect and Watcher, 2026-09-19, on the local machine (i7-10700K, 32 GB
RAM, GTX 1660 Ti). Started 20:05 on a freshly started Ollama with `ctx_bench.py`.
The run was stopped at 21:11 by Claude Code, not by the script, because the
system was critically low on memory while `nemotron-3.5-lightning` (25 GB) was
loading. Two of five tests finished; nemotron, the function agent and the urge
agent were not run.

| Test | Prompt tokens | Load | Prefill | Generation | Reply |
|---|---|---|---|---|---|
| qwen3.8:27b (sable prompt) | 5,904 | 52.6 s | 447.5 s (13.2 tok/s) | 3,000 tokens in 2,427 s (1.24 tok/s), stopped at the cap | 0 characters: 14,034 characters of thinking used the whole budget |
| muse-glimmer:30b (quill prompt) | 5,811 | 57.0 s | 350.4 s (16.6 tok/s) | 965 tokens in 615 s (1.57 tok/s), finished normally | 978 characters, after 3,968 characters of thinking |

Whole calls: 2,929 s (48.8 min) for qwen3.8 and 1,025 s (17.1 min) for muse-glimmer.

Memory: free RAM went from 22.6 GB to 7.4 GB with qwen3.8 loaded (18.9 GB, only
2.5 GB in VRAM), and muse-glimmer was 17.5 GB (3.0 GB in VRAM).

The full thinking and replies are in `voice-sable.txt` and `voice-quill.txt`.
Per-call numbers are in `results.jsonl`.
