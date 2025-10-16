import json
import logging
import os
import asyncio
from pathlib import Path
from datetime import datetime
import threading
import time
import hashlib
import colorsys
import math
from typing import Optional, Callable
import tkinter as tk
from tkinter import scrolledtext, simpledialog, filedialog, messagebox
from tkinter import ttk

import discord
import requests

from config import CONF_DIR, REQUIRED_CONFIGS
from config.checks import check_required_configs
from config_loader import (
    load_globals,
    try_load_globals,
    try_load_pdvs,
    try_load_classes,
    try_load_agents,
    save_globals,
    save_pdvs,
    save_classes,
    save_agents,
)
from pdv_utils import apply_and_persist_pdv_adjustments

logger = logging.getLogger(__name__)

CHATLOG_DIR = "chatlogs"
SENT_MESSAGES_PATH = os.path.join(CHATLOG_DIR, "messages_to_humans.json")

# Special entry shown at the top of group-pick lists (Agent mode)
_NEW_GROUP_LABEL = "***Create New Group***"

_discord_queue: asyncio.Queue | None = None
_discord_client: discord.Client | None = None
_discord_task: asyncio.Task | None = None
_discord_consumer_task: asyncio.Task | None = None
_discord_loop: asyncio.AbstractEventLoop | None = None


def _get_conductor():
    """Return the *running* conductor module even if it was launched as a script."""

    import sys
    import importlib
    import os

    module = sys.modules.get("conductor")
    if module is not None:
        return module

    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", "")
    if main_file and os.path.basename(main_file) == "conductor.py":
        return main

    return importlib.import_module("conductor")

def get_discord_queue() -> asyncio.Queue:
    global _discord_queue
    if _discord_queue is None:
        _discord_queue = asyncio.Queue()
    return _discord_queue


class _DiscordInUI(discord.Client):
    async def on_ready(self):
        print(f"[UI/Discord] Logged in as {self.user}")

    async def on_message(self, msg):
        channel_id_env = os.getenv("DISCORD_CHANNEL_ID")
        try:
            target_channel = int(channel_id_env) if channel_id_env else 0
        except Exception:
            target_channel = 0

        if msg.author.bot or (target_channel and msg.channel.id != target_channel):
            return

        author = getattr(msg.author, "display_name", str(msg.author))
        entry = {
            "source": "discord",
            "author": author,
            "text": msg.content,
            "timestamp": msg.created_at.isoformat(),
        }

        await get_discord_queue().put(entry)

        try:
            g = load_globals() or {}
            adjs = g.get("incoming_message_pdvms") or g.get("incoming_message_dpvms") or []
            if isinstance(adjs, list) and adjs:
                norm = []
                for item in adjs:
                    if "delta_pct" in item and "delta" not in item:
                        try:
                            pct = float(item["delta_pct"])
                            item = {**item, "delta": pct / 100.0}
                            item.pop("delta_pct", None)
                        except Exception:
                            continue
                    norm.append(item)
                if norm:
                    apply_and_persist_pdv_adjustments(norm)
                    # Best-effort push to keep conductor in sync immediately.
                    try:
                        c = _get_conductor()
                        if hasattr(c, "_refresh_pdvs_from_disk"):
                            c._refresh_pdvs_from_disk()
                    except Exception:
                        pass
                    print("[PDVM] Applied incoming_message_pdvms on Discord message.")
        except Exception as e:
            print(f"[PDVM] Failed applying incoming_message_pdvms: {e}")


async def _discord_consumer_loop():
    q = get_discord_queue()
    while True:
        item = await q.get()
        try:
            c = _get_conductor()
            if hasattr(c, "inject_external_message"):
                await c.inject_external_message(item.get("text", ""), item)
            else:
                await c.handle_user_message(item.get("text", ""), meta=item)
        except Exception as e:
            print(f"[UI/Discord] Consumer error: {e}")
        finally:
            q.task_done()



async def start_discord_in_ui():
    global _discord_client, _discord_task, _discord_consumer_task, _discord_loop

    token = os.getenv("fenra_token")
    channel_id = os.getenv("DISCORD_CHANNEL_ID")
    if not token or not channel_id:
        print("[UI/Discord] Disabled (missing fenra_token or DISCORD_CHANNEL_ID).")
        return

    intents = discord.Intents.default()
    intents.message_content = True
    _discord_client = _DiscordInUI(intents=intents)

    _discord_loop = asyncio.get_running_loop()
    loop = _discord_loop
    # Queue removed: do not start consumer loop.

    async def _runner():
        try:
            await _discord_client.start(token)
        except Exception as e:
            print(f"[UI/Discord] Client stopped: {e}")

    _discord_task = loop.create_task(_runner())
    print("[UI/Discord] Listening started.")


async def stop_discord_in_ui():
    global _discord_client, _discord_task, _discord_consumer_task
    if _discord_client:
        try:
            await _discord_client.close()
        except Exception:
            pass
    if _discord_task:
        _discord_task.cancel()
    if _discord_consumer_task:
        _discord_consumer_task.cancel()
    print("[UI/Discord] Listening stopped.")

# ─── public: fetch recent discord messages for listeners ──────────────────────
async def _discord_fetch_recent(n: int) -> list[dict]:
    """Coroutine to fetch last n messages from configured channel."""
    if _discord_client is None:
        return []
    try:
        chan_id = int(os.getenv("DISCORD_CHANNEL_ID") or "0")
    except Exception:
        chan_id = 0
    if not chan_id:
        return []
    ch = _discord_client.get_channel(chan_id)
    if ch is None:
        try:
            ch = await _discord_client.fetch_channel(chan_id)
        except Exception:
            return []
    out: list[dict] = []
    async for m in ch.history(limit=int(n)):
        out.append({
            "author": getattr(m.author, "display_name", str(m.author)),
            "text": m.content,
            "timestamp": m.created_at.isoformat(),
        })
    return out

def fetch_recent_discord_messages(n: int = 10) -> list[dict]:
    """Sync wrapper to retrieve last n messages from Discord."""
    loop = _discord_loop
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return []
    fut = asyncio.run_coroutine_threadsafe(_discord_fetch_recent(int(n)), loop)
    try:
        return fut.result(timeout=5)
    except Exception:
        return []


def hsl_to_hex(h: int, s: float, l: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def pastel_for_class(name: str) -> str:
    h = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % 360
    return hsl_to_hex(h, 0.45, 0.82)


class FenraUI:
    """Simple UI for displaying output and listing AIs."""

    def __init__(
        self,
        agents,
        inject_callback=None,
        send_callback=None,
        config_path="confs/globals.json",
        on_apply_globals: Optional[Callable[[dict], None]] = None,
    ):
        logger.debug(
            "Entering FenraUI.__init__ with agents=%s inject_callback=%s send_callback=%s",
            agents,
            inject_callback,
            send_callback,
        )
        self.root = tk.Tk()
        self.root.title("Fenra")
        self.agents = agents
        self.inject_callback = inject_callback
        self.send_callback = send_callback
        self.config_path = config_path
        self.on_apply_globals = on_apply_globals

        self.sent_messages = []
        self.log_messages = []
        self._agent_payloads: dict[str, str] = {}
        self._active_agent: Optional[str] = None
        self._group_contexts: dict[str, str] = {}

        self._model_cache: list[str] = []
        self._is_running = False
        self._conf_dir_path = Path(CONF_DIR).resolve()
        self._conf_dir = str(self._conf_dir_path)
        # Pre-create Live Metrics containers so early rebuilds are safe
        self.metric_bars: dict[str, ttk.Progressbar] = {}
        self.metric_labels: dict[str, tk.Label] = {}
        self.metrics_rows: ttk.Frame | None = None
        self.metrics_canvas: tk.Canvas | None = None
        self.metrics_legend: ttk.Frame | None = None
        self._conf_presence: dict[str, bool] = {}
        self._config_watch_stop = threading.Event()
        self._config_watch_thread: threading.Thread | None = None
        self._config_observer = None
        self._config_update_pending = False
        self._missing_configs: list[str] = []

        self._classes_map: dict[str, dict] = self._ui_classes()
        self._pdv_names: list[str] = sorted(self._ui_pdvs().keys())
        # drop adjustments for PDVs that no longer exist
        for c in self._classes_map.values():
            if "pdv_adjustments" in c:
                c["pdv_adjustments"] = [
                    a for a in c.get("pdv_adjustments", []) if a.get("name") in self._pdv_names
                ]
        self._classes_dirty = False
        self._loading_class = False

        self.global_config = self._ui_globals()

        pdv_cfg = self._ui_pdvs()
        self.pdv_values = {name: cfg.get("value", 0.0) for name, cfg in pdv_cfg.items()}

        self._agents_model: list[dict] = self._ui_agents()
        self._active_agent_index: Optional[int] = None
        self._agent_form_vars: dict[str, tk.Variable] = {}
        self._agent_text_fields: dict[str, scrolledtext.ScrolledText] = {}
        self.agent_list: tk.Listbox | None = None
        self._agent_class_cb: ttk.Combobox | None = None
        self._agent_model_hint: ttk.Label | None = None
        self._loading_agent_form = False
        self._pdv_row_widgets: list[tuple[ttk.Combobox, tk.DoubleVar, ttk.Scale, ttk.Button]] = []

        # ── Run Control Toolbar ───────────────────────────────────────────────
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, pady=(4, 0))
        self._run_btn_text = tk.StringVar(value="Start")
        self._run_status = tk.StringVar(value="Idle")
        run_controls = ttk.Frame(toolbar)
        run_controls.pack(side=tk.LEFT, padx=4)
        self._run_button = ttk.Button(
            run_controls, textvariable=self._run_btn_text, command=self._toggle_run
        )
        self._run_button.pack()
        self._run_hint_var = tk.StringVar(value="")
        self._run_hint_label = ttk.Label(
            run_controls,
            textvariable=self._run_hint_var,
            foreground="#b94a48",
            wraplength=220,
            justify=tk.LEFT,
        )
        ttk.Label(toolbar, textvariable=self._run_status).pack(side=tk.LEFT, padx=8)
        self._config_refresh_btn = ttk.Button(
            toolbar, text="Refresh configs", command=self._manual_refresh_required_configs
        )
        self._config_refresh_btn.pack(side=tk.LEFT, padx=4)

        # Try to reflect initial state from conductor (if available)
        try:
            c = _get_conductor()
            if hasattr(c, "is_processing") and c.is_processing():
                self._run_btn_text.set("Stop")
                self._run_status.set("Running")
                self._is_running = True
        except Exception:
            pass

        # ── Main Notebook ────────────────────────────────────────────────────
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ----- Configurations Tab -----
        configs_tab = ttk.Frame(self.notebook)
        self.notebook.add(configs_tab, text="Configurations")

        self.config_nb = ttk.Notebook(configs_tab)
        self.config_nb.pack(fill=tk.BOTH, expand=True)

        # Globals sub-tab
        self.globals_tab = ttk.Frame(self.config_nb)
        self.config_nb.add(self.globals_tab, text="Globals")
        self._build_globals_tab()

        # PDVs sub-tab
        self.pdvs_tab = ttk.Frame(self.config_nb)
        self.config_nb.add(self.pdvs_tab, text="PDVs")
        self._build_pdvs_tab()

        # Agent Classes sub-tab
        self.classes_tab = ttk.Frame(self.config_nb)
        self.config_nb.add(self.classes_tab, text="Agent Classes")
        self._build_classes_tab()
        # persist and restore last selection
        last = self._load_ui_state().get("last_class")
        if last and last in self._classes_map:
            items = list(self.cls_list.get(0, tk.END))
            if last in items:
                idx = items.index(last)
                self.cls_list.selection_set(idx)
        elif self.cls_list.size() > 0:
            self.cls_list.selection_set(0)
        if self.cls_list.size() > 0:
            self._on_class_select()
        # prompt on tab change if dirty
        self.config_nb.bind("<<NotebookTabChanged>>", self._on_config_tab_changed)
        # mark dirty on text edits
        for st in (self.cls_sys, self.cls_pre, self.cls_post):
            st.bind("<KeyRelease>", self._mark_classes_dirty_from_text)

        # Agents sub-tab
        self.agents_editor_tab = ttk.Frame(self.config_nb)
        self.config_nb.add(self.agents_editor_tab, text="Agents")
        self._build_agents_editor_tab()

        # Groups sub-tab
        self.simple_groups_tab = ttk.Frame(self.config_nb)
        self.config_nb.add(self.simple_groups_tab, text="Groups")
        self._ensure_agent_group_membership()
        self._build_simple_groups_tab()

        # ----- Live Metrics Tab -----
        self.metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_tab, text="Live Metrics")

        # container that holds the pie and a legend
        self.metrics_rows = ttk.Frame(self.metrics_tab)  # reuse name to minimize ripple
        self.metrics_rows.pack(fill=tk.BOTH, expand=True)
        self._rebuild_live_metrics_rows()

        limit = self.global_config.get("max_context_tokens", 8192)
        self.token_usage_var = tk.StringVar(value=f"Tokens: 0 / {limit}")
        tk.Label(self.metrics_tab, textvariable=self.token_usage_var).pack(anchor="w", padx=4, pady=2)

        # ----- Internal Thoughts Tab -----
        thoughts_tab = ttk.Frame(self.notebook)
        self.notebook.add(thoughts_tab, text="Internal Thoughts")

        paned = ttk.Panedwindow(thoughts_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        thought_frame = ttk.Frame(paned)
        event_frame = ttk.Frame(paned)
        paned.add(thought_frame, weight=1)
        paned.add(event_frame, weight=1)

        self.thought_stream = scrolledtext.ScrolledText(thought_frame, state="disabled")
        self.thought_stream.pack(fill=tk.BOTH, expand=True)

        self.events_stream = scrolledtext.ScrolledText(event_frame, state="disabled")
        self.events_stream.pack(fill=tk.BOTH, expand=True)

        # Backward compatibility
        self.output = self.thought_stream

        self.base_timeout = self.global_config.get("watchdog_timeout", 900)
        label_txt = (
            "Base Timeout: disabled"
            if (self.base_timeout is None or float(self.base_timeout) <= 0)
            else f"Base Timeout: {int(self.base_timeout)}s"
        )
        self.timeout_label = ttk.Label(thoughts_tab, text=label_txt)
        self.timeout_label.pack(anchor="w", padx=4, pady=2)

        self._refresh_log_display()

        # ----- Messages Tab -----
        messages_tab = ttk.Frame(self.notebook)
        self.notebook.add(messages_tab, text="Messages")

        msg_top = ttk.Frame(messages_tab)
        msg_top.pack(fill=tk.X, pady=2)
        ttk.Button(msg_top, text="Refresh", command=self.update_queue_and_sent).pack(
            side=tk.RIGHT, padx=2
        )

        queued_frame = ttk.LabelFrame(messages_tab, text="Queued (from humans)")
        queued_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.queued_text = scrolledtext.ScrolledText(queued_frame, state="disabled", height=10)
        self.queued_text.pack(fill=tk.BOTH, expand=True)

        sent_frame = ttk.LabelFrame(messages_tab, text="Sent (to humans)")
        sent_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.sent_text = scrolledtext.ScrolledText(sent_frame, state="disabled", height=10)
        self.sent_text.pack(fill=tk.BOTH, expand=True)

        self.update_queue_and_sent()

        # ----- Topology Tab -----
        topology_tab = ttk.Frame(self.notebook)
        self.notebook.add(topology_tab, text="Topology")

        self.topology_header = ttk.Label(topology_tab, text="Active Agent: None")
        self.topology_header.pack(anchor="w", padx=4, pady=2)

        self.topology_canvas = tk.Canvas(topology_tab, background="white")
        self.topology_canvas.pack(fill=tk.BOTH, expand=True)
        self.topology_canvas.bind("<Configure>", lambda e: self._redraw_topology())

        self._topology_active = None
        self._topology_agents = []
        self._topology_node_items = {}
        self._topology_tooltip = None

        # ----- Agent Context Tab -----
        agent_tab = ttk.Frame(self.notebook)
        self.notebook.add(agent_tab, text="Agent Context")

        self.agent_header = ttk.Label(agent_tab, text="Current Agent: None")
        self.agent_header.pack(anchor="w", padx=4, pady=2)

        agent_toolbar = ttk.Frame(agent_tab)
        agent_toolbar.pack(anchor="w", padx=4, pady=2)
        ttk.Button(agent_toolbar, text="Copy JSON", command=self._copy_agent_payload).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(agent_toolbar, text="Save…", command=self._save_agent_payload).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(agent_toolbar, text="Clear", command=self._clear_agent_payload).pack(
            side=tk.LEFT, padx=2
        )

        self.agent_payload_view = scrolledtext.ScrolledText(
            agent_tab, state="disabled"
        )
        self.agent_payload_view.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # ----- Group Context Tab -----
        group_tab = ttk.Frame(self.notebook)
        self.notebook.add(group_tab, text="Group Context")

        group_paned = ttk.Panedwindow(group_tab, orient=tk.HORIZONTAL)
        group_paned.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.Frame(group_paned)
        text_frame = ttk.Frame(group_paned)
        group_paned.add(list_frame, weight=1)
        group_paned.add(text_frame, weight=3)

        self.group_list = tk.Listbox(list_frame, exportselection=False)
        self.group_list.pack(fill=tk.BOTH, expand=True)
        self.group_list.bind("<<ListboxSelect>>", self._on_group_select)

        self.group_text = scrolledtext.ScrolledText(text_frame, state="disabled")
        self.group_text.pack(fill=tk.BOTH, expand=True)

        self.update_pdvs(self.pdv_values)
        self._start_metrics_poll()
        # Do not auto-open the Globals modal. Users will set values in the Globals tab.

        os.makedirs(self._conf_dir, exist_ok=True)
        self._start_config_watcher()
        self._update_required_configs_state()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        logger.debug("Exiting FenraUI.__init__")

    def _toggle_run(self):
        """Start/Stop the agent processing loop in conductor."""

        if not self._is_running:
            all_present, missing = check_required_configs(self._conf_dir)
            if not all_present:
                self._apply_required_configs_state(all_present, missing)
                messagebox.showwarning(
                    "Missing Config Files",
                    "Cannot start until these configs exist in"
                    f" {CONF_DIR}/: {', '.join(missing)}",
                )
                return

        try:
            c = _get_conductor()
        except Exception as e:
            messagebox.showerror("Run Control Error", f"Failed to import conductor: {e}")
            return

        print(f"[UI] Toggling run on module id={id(c)} file={getattr(c, '__file__', None)}")

        try:
            running = False
            if hasattr(c, "is_processing"):
                running = bool(c.is_processing())
        except Exception:
            running = False

        try:
            if running and hasattr(c, "stop_processing"):
                c.stop_processing()
                self._run_btn_text.set("Start")
                self._run_status.set("Paused")
                self._is_running = False
                self._update_required_configs_state()
            elif not running and hasattr(c, "start_processing"):
                c.start_processing()
                self._run_btn_text.set("Stop")
                self._run_status.set("Running")
                self._is_running = True
                self._missing_configs = []
                self._apply_required_configs_state(True, [])
            else:
                messagebox.showerror("Run Control Error", "Conductor run control not available.")
        except FileNotFoundError:
            all_present, missing = check_required_configs(self._conf_dir)
            self._run_btn_text.set("Start")
            self._run_status.set("Paused")
            self._is_running = False
            self._missing_configs = missing
            self._apply_required_configs_state(all_present, missing)
            messagebox.showwarning(
                "Missing Config Files",
                "Cannot start until these configs exist in"
                f" {CONF_DIR}/: {', '.join(missing)}",
            )
        except Exception as e:
            messagebox.showerror("Run Control Error", str(e))


    def _manual_refresh_required_configs(self) -> None:
        logger.debug("Manual refresh of required configs requested")
        self._update_required_configs_state()

    def _schedule_required_configs_update(self) -> None:
        if self._config_update_pending:
            return

        def _run():
            self._config_update_pending = False
            self._update_required_configs_state()

        self._config_update_pending = True
        try:
            self.root.after(150, _run)
        except tk.TclError:
            self._config_update_pending = False

    def _update_required_configs_state(self) -> None:
        # Compute presence strictly by existence on disk.
        presence_changes: list[tuple[str, bool]] = []
        missing: list[str] = []
        for name in REQUIRED_CONFIGS:
            present = self._have_conf(name)
            if self._conf_presence.get(name) != present:
                presence_changes.append((name, present))
            self._conf_presence[name] = present
            if not present:
                missing.append(name)

        self._missing_configs = missing
        all_present = len(missing) == 0
        self._apply_required_configs_state(all_present, missing)
        if presence_changes:
            self._handle_conf_presence_changes(presence_changes)

    def _apply_required_configs_state(self, all_present: bool, missing: list[str]) -> None:
        hint = ""
        if not all_present:
            hint = f"Missing config files in {CONF_DIR}/: {', '.join(missing)}"
        self._show_run_hint(hint)
        if all_present or self._is_running:
            self._run_button.state(["!disabled"])
        else:
            self._run_button.state(["disabled"])

    def _show_run_hint(self, text: str) -> None:
        self._run_hint_var.set(text)
        if text:
            if not self._run_hint_label.winfo_ismapped():
                self._run_hint_label.pack(fill=tk.X, pady=(2, 0))
        else:
            if self._run_hint_label.winfo_ismapped():
                self._run_hint_label.pack_forget()

    def _start_config_watcher(self) -> None:
        logger.debug("Starting configuration watcher for %s", self._conf_dir)
        self._config_watch_stop.clear()
        try:
            from watchdog.events import FileSystemEventHandler  # type: ignore
            from watchdog.observers import Observer  # type: ignore
        except Exception:
            logger.debug("watchdog not available; falling back to polling")
            self._config_watch_thread = threading.Thread(
                target=self._poll_config_dir, args=(self._conf_dir,), daemon=True
            )
            self._config_watch_thread.start()
            return

        class _ConfigHandler(FileSystemEventHandler):
            def __init__(self, outer: "FenraUI") -> None:
                self._outer = outer

            def on_any_event(self, event) -> None:  # type: ignore[override]
                if getattr(event, "is_directory", False):
                    return
                self._outer._schedule_required_configs_update()

        observer = Observer()
        try:
            observer.schedule(_ConfigHandler(self), self._conf_dir, recursive=False)
            observer.start()
            self._config_observer = observer
        except Exception as exc:
            logger.warning("Failed to start watchdog observer: %s", exc)
            try:
                observer.stop()
                observer.join(timeout=1)
            except Exception:
                pass
            self._config_observer = None
            self._config_watch_thread = threading.Thread(
                target=self._poll_config_dir, args=(self._conf_dir,), daemon=True
            )
            self._config_watch_thread.start()

    def _poll_config_dir(self, conf_dir: str) -> None:
        last_missing: tuple[str, ...] | None = None
        while not self._config_watch_stop.wait(1.0):
            _, missing = check_required_configs(conf_dir)
            state = tuple(missing)
            if state != last_missing:
                last_missing = state
                self._schedule_required_configs_update()

    def _stop_config_watcher(self) -> None:
        self._config_watch_stop.set()
        if self._config_observer is not None:
            try:
                self._config_observer.stop()
                self._config_observer.join(timeout=2.0)
            except Exception:
                pass
            self._config_observer = None
        if self._config_watch_thread is not None and self._config_watch_thread.is_alive():
            self._config_watch_thread.join(timeout=2.0)
        self._config_watch_thread = None

    def _on_close(self) -> None:
        self._stop_config_watcher()
        try:
            self.root.destroy()
        except Exception:
            pass


    class _InjectDialog(simpledialog.Dialog):
        """Dialog for entering a message to inject."""

        def __init__(self, parent, group_name: str):
            logger.debug("Entering _InjectDialog.__init__ group_name=%s", group_name)
            self.group_name = group_name
            self.message = ""
            super().__init__(parent, title="Inject Message")
            logger.debug("Exiting _InjectDialog.__init__")

        def body(self, master):
            logger.debug("Entering _InjectDialog.body")
            tk.Label(master, text=f"Send message to {self.group_name}:").grid(row=0, column=0, sticky="w")
            self.text = scrolledtext.ScrolledText(master, width=40, height=10)
            self.text.grid(row=1, column=0, sticky="nsew")
            master.grid_rowconfigure(1, weight=1)
            master.grid_columnconfigure(0, weight=1)
            logger.debug("Exiting _InjectDialog.body")
            return self.text

        def buttonbox(self):
            box = tk.Frame(self)
            send = tk.Button(box, text="Send", width=10, command=self.ok, default=tk.ACTIVE)
            send.pack(side=tk.LEFT, padx=5, pady=5)
            cancel = tk.Button(box, text="Cancel", width=10, command=self.cancel)
            cancel.pack(side=tk.LEFT, padx=5, pady=5)
            self.bind("<Escape>", self.cancel)
            box.pack()

        def apply(self):
            logger.debug("Entering _InjectDialog.apply")
            self.message = self.text.get("1.0", tk.END).rstrip()
            self.result = self.message
            logger.debug("Exiting _InjectDialog.apply")

    class _SendDialog(simpledialog.Dialog):
        """Dialog for entering a message for the listeners."""

        def body(self, master):
            logger.debug("Entering _SendDialog.body")
            tk.Label(master, text="Message to user:").grid(row=0, column=0, sticky="w")
            self.text = scrolledtext.ScrolledText(master, width=40, height=10)
            self.text.grid(row=1, column=0, sticky="nsew")
            tk.Label(master, text="Groups (comma-separated):").grid(row=2, column=0, sticky="w")
            self.groups_entry = tk.Entry(master)
            self.groups_entry.grid(row=3, column=0, sticky="ew")
            master.grid_rowconfigure(1, weight=1)
            master.grid_columnconfigure(0, weight=1)
            logger.debug("Exiting _SendDialog.body")
            return self.text

        def buttonbox(self):
            box = tk.Frame(self)
            send = tk.Button(box, text="Send", width=10, command=self.ok, default=tk.ACTIVE)
            send.pack(side=tk.LEFT, padx=5, pady=5)
            cancel = tk.Button(box, text="Cancel", width=10, command=self.cancel)
            cancel.pack(side=tk.LEFT, padx=5, pady=5)
            self.bind("<Escape>", self.cancel)
            box.pack()

        def apply(self):
            logger.debug("Entering _SendDialog.apply")
            self.message = self.text.get("1.0", tk.END).rstrip()
            groups_text = self.groups_entry.get().strip()
            groups = [g.strip() for g in groups_text.split(",") if g.strip()]
            self.result = {"message": self.message, "groups": groups}
        logger.debug("Exiting _SendDialog.apply")

    def _inject_message(self):
        logger.debug("Entering _inject_message")
        group_name = "All Groups"
        dialog = self._InjectDialog(self.root, group_name)
        result = dialog.result
        if result:
            if self.inject_callback:
                self.inject_callback(group_name, result)
            else:
                items = self._enqueue_message("system", result)
                self.update_queue(items)
                try:
                    c = _get_conductor()
                    import asyncio
                    asyncio.run(c.inject_external_message(result, {"author": "system"}))
                except Exception as e:
                    print(f"[UI] inject_external_message failed: {e}")
        logger.debug("Exiting _inject_message")

    def _send_message(self):
        logger.debug("Entering _send_message")
        dialog = self._SendDialog(self.root)
        result = dialog.result
        if result:
            groups = result.get("groups") if isinstance(result, dict) else []
            if not groups:
                messagebox.showerror("Error", "Please specify at least one group")
            else:
                if self.send_callback:
                    self.send_callback(result["message"], groups)
                else:
                    items = self._enqueue_message("user", result["message"])
                    self.update_queue(items)
                    try:
                        c = _get_conductor()
                        import asyncio
                        asyncio.run(
                            c.inject_external_message(result["message"], {"author": "user", "groups": groups})
                        )
                    except Exception as e:
                        print(f"[UI] inject_external_message failed: {e}")
        logger.debug("Exiting _send_message")

    def update_queue(self, messages):
        logger.debug("Entering update_queue messages=%s", messages)
        self.update_queue_and_sent(queued=messages)
        logger.debug("Exiting update_queue")

    def update_sent(self, messages):
        logger.debug("Entering update_sent messages=%s", messages)
        self.sent_messages = list(messages)
        self.update_queue_and_sent(sent=messages)
        logger.debug("Exiting update_sent")

    def update_pdvs(self, pdv_values: dict[str, float]) -> None:
        logger.debug("Entering update_pdvs pdv_values=%s", pdv_values)

        def _update():
            # Store and redraw pie
            self.pdv_values = dict(pdv_values or {})
            self._draw_pdv_pie(self.pdv_values)

        self._threadsafe(_update)
        logger.debug("Exiting update_pdvs")

    def set_token_usage(self, used: int, limit: int) -> None:
        def _update():
            self.token_usage_var.set(f"Tokens: {used} / {limit}")
        self._threadsafe(_update)

    def append_thought(self, text: str, timestamp: Optional[str] = None) -> None:
        logger.debug("Entering append_thought text=%s timestamp=%s", text, timestamp)

        def _append():
            ts = timestamp or time.strftime("%H:%M:%S")
            self._append_text(self.thought_stream, f"[{ts}] {text}\n")

        self._threadsafe(_append)
        logger.debug("Exiting append_thought")

    def append_event(self, text: str, timestamp: Optional[str] = None) -> None:
        logger.debug("Entering append_event text=%s timestamp=%s", text, timestamp)

        def _append():
            ts = timestamp or time.strftime("%H:%M:%S")
            self._append_text(self.events_stream, f"[{ts}] {text}\n")

        self._threadsafe(_append)
        logger.debug("Exiting append_event")

    def update_queue_and_sent(self, queued: Optional[list] = None, sent: Optional[list] = None) -> None:
        logger.debug("Entering update_queue_and_sent queued=%s sent=%s", queued, sent)

        def _load(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                return None

        if queued is None:
            try:
                c = _get_conductor()
                queued = getattr(c, "_INCOMING_QUEUE", [])
            except Exception:
                queued = []
        if sent is None:
            sent = _load(SENT_MESSAGES_PATH)

        def _render():
            # Queued messages
            self.queued_text.configure(state="normal")
            self.queued_text.delete("1.0", tk.END)
            if isinstance(queued, list) and queued:
                for entry in queued:
                    if isinstance(entry, dict):
                        ts = entry.get("timestamp", "unknown")
                        msg = entry.get("raw_message") or entry.get("message") or str(entry)
                    else:
                        ts = "unknown"
                        msg = str(entry)
                    self.queued_text.insert(tk.END, f"[{ts}] {msg}\n")
            else:
                self.queued_text.insert(tk.END, "No queued messages.\n")
            self.queued_text.configure(state="disabled")
            self.queued_text.see(tk.END)

            # Sent messages
            self.sent_text.configure(state="normal")
            self.sent_text.delete("1.0", tk.END)
            if isinstance(sent, list) and sent:
                for entry in sent:
                    if isinstance(entry, dict):
                        ts = entry.get("timestamp", "unknown")
                        sender = entry.get("sender", "unknown")
                        msg = entry.get("message", "")
                    else:
                        ts = "unknown"
                        sender = "unknown"
                        msg = str(entry)
                    self.sent_text.insert(tk.END, f"[{ts}] {sender}: {msg}\n")
            else:
                self.sent_text.insert(tk.END, "No sent messages.\n")
            self.sent_text.configure(state="disabled")
            self.sent_text.see(tk.END)

        self._threadsafe(_render)
        logger.debug("Exiting update_queue_and_sent")

    # ------------------------------------------------------------------
    # Configuration tab builders
    # ------------------------------------------------------------------

    def _build_globals_tab(self) -> None:
        for child in self.globals_tab.winfo_children():
            child.destroy()
        self._globals_vars = {}
        self.global_config = self._ui_globals()
        models = self._fetch_models()
        self._globals_vars = {
            "debug_level": tk.StringVar(value=self.global_config.get("debug_level", "INFO")),
            "model": tk.StringVar(value=self.global_config.get("model", "")),
            "temperature": tk.StringVar(value=str(self.global_config.get("temperature", ""))),
            "system_prompt": tk.StringVar(value=self.global_config.get("system_prompt", "")),
            "pre_context_message": tk.StringVar(value=self.global_config.get("pre_context_message", "")),
            "post_context_message": tk.StringVar(value=self.global_config.get("post_context_message", "")),
            "max_context_tokens": tk.StringVar(value=str(self.global_config.get("max_context_tokens", 8192))),
            "pdv_gamma": tk.StringVar(value=str(self.global_config.get("pdv_gamma", 2.0))),
            "watchdog_timeout": tk.StringVar(value=str(self.global_config.get("watchdog_timeout", 900))),
        }
        row = 0
        for label, key in [
            ("Debug Level", "debug_level"),
            ("Model", "model"),
            ("Temperature", "temperature"),
            ("System Prompt", "system_prompt"),
            ("Pre Context", "pre_context_message"),
            ("Post Context", "post_context_message"),
            ("Max Tokens", "max_context_tokens"),
            ("PDV Gamma", "pdv_gamma"),
            ("Watchdog Timeout (s)  (0=disabled)", "watchdog_timeout"),
        ]:
            tk.Label(self.globals_tab, text=label).grid(row=row, column=0, sticky="w")
            if key == "model":
                box = ttk.Combobox(
                    self.globals_tab,
                    textvariable=self._globals_vars[key],
                    values=models,
                    state="readonly",
                )
                box.grid(row=row, column=1, sticky="ew")
            else:
                entry = ttk.Entry(self.globals_tab, textvariable=self._globals_vars[key])
                entry.grid(row=row, column=1, sticky="ew")
            row += 1
        self.globals_tab.columnconfigure(1, weight=1)
        btn_frame = ttk.Frame(self.globals_tab)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=4)
        ttk.Button(btn_frame, text="Refresh Models", command=self._refresh_models).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Save", command=self._save_globals).pack(side=tk.LEFT, padx=2)
        self.base_timeout = self.global_config.get("watchdog_timeout", 900)
        if hasattr(self, "timeout_label"):
            txt = (
                "Base Timeout: disabled"
                if (self.base_timeout is None or float(self.base_timeout) <= 0)
                else f"Base Timeout: {int(self.base_timeout)}s"
            )
            self.timeout_label.config(text=txt)

    def _refresh_models(self) -> None:
        models = self._fetch_models()
        box = None
        for child in self.globals_tab.grid_slaves():
            if isinstance(child, ttk.Combobox):
                box = child
                break
        if box is not None:
            box.configure(values=models)

    def _save_globals(self) -> None:
        for k, var in self._globals_vars.items():
            val = var.get()
            if k in {"temperature", "max_context_tokens", "pdv_gamma", "watchdog_timeout"}:
                try:
                    if k == "max_context_tokens":
                        self.global_config[k] = int(val)
                    elif k == "watchdog_timeout":
                        self.global_config[k] = int(float(val))
                    else:
                        self.global_config[k] = float(val)
                except ValueError:
                    self.global_config[k] = None
            else:
                self.global_config[k] = val
        save_globals(self.global_config)
        self._update_required_configs_state()
        # refresh label
        self.base_timeout = self.global_config.get("watchdog_timeout", 900)
        txt = (
            "Base Timeout: disabled"
            if (self.base_timeout is None or float(self.base_timeout) <= 0)
            else f"Base Timeout: {int(self.base_timeout)}s"
        )
        self.timeout_label.config(text=txt)
        # Live-apply to the running Conductor
        try:
            if self.on_apply_globals:
                self.on_apply_globals(dict(self.global_config))
        except Exception:
            logger.exception("Failed to apply globals update callback")

    def _build_pdvs_tab(self) -> None:
        for child in self.pdvs_tab.winfo_children():
            child.destroy()
        self._pdv_rows = []
        pdvs = self._ui_pdvs()
        self.pdv_values = {name: cfg.get("value", 0.0) for name, cfg in pdvs.items()}
        self._pdv_names = sorted(pdvs.keys())
        frame = self.pdvs_tab
        for name, cfg in pdvs.items():
            self._add_pdv_row(frame, name, cfg)
        btn = ttk.Frame(frame)
        btn.pack(fill=tk.X, pady=4)
        ttk.Button(
            btn,
            text="Add",
            command=lambda: self._add_pdv_row(frame, "new", {"description": "", "value": 0.0}),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn, text="Save", command=self._save_pdvs).pack(side=tk.LEFT, padx=2)
        if getattr(self, "metrics_rows", None):
            self._rebuild_live_metrics_rows()
        self._refresh_class_pdv_choices()

    def _add_pdv_row(self, parent, name, cfg):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=4, pady=2)
        name_var = tk.StringVar(value=name)
        desc_var = tk.StringVar(value=cfg.get("description", ""))
        val_var = tk.StringVar(value=str(cfg.get("value", 0.0)))

        ttk.Entry(row, textvariable=name_var, width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=desc_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Entry(row, textvariable=val_var, width=10).pack(side=tk.LEFT, padx=2)

        ttk.Button(row, text="Remove", command=lambda r=row: r.destroy()).pack(side=tk.LEFT, padx=2)
        # store the row parts (note: val_var is now StringVar)
        self._pdv_rows.append((row, name_var, desc_var, val_var))

    def _save_pdvs(self) -> None:
        data = {}
        for row, name_var, desc_var, val_var in self._pdv_rows:
            if not row.winfo_exists():
                continue
            name = name_var.get().strip()
            if not name:
                continue
            try:
                v = float(str(val_var.get()).strip())
            except Exception:
                v = 0.0
            if v < 0.0:
                v = 0.0
            data[name] = {
                "name": name,
                "description": desc_var.get(),
                "value": v,
            }
        save_pdvs(data)
        self._update_required_configs_state()
        self.pdv_values = {n: cfg["value"] for n, cfg in data.items()}
        self._pdv_names = sorted(data.keys())
        for cls in self._classes_map.values():
            if "pdv_adjustments" in cls:
                cls["pdv_adjustments"] = [
                    a for a in cls["pdv_adjustments"] if a.get("name") in self._pdv_names
                ]
        self._refresh_class_pdv_choices()
        self._rebuild_live_metrics_rows()
        self.update_pdvs(self.pdv_values)

    def _build_classes_tab(self) -> None:
        for child in self.classes_tab.winfo_children():
            child.destroy()
        self._classes_map = self._ui_classes()
        for c in self._classes_map.values():
            if "pdv_adjustments" in c:
                c["pdv_adjustments"] = [
                    a for a in c.get("pdv_adjustments", []) if a.get("name") in self._pdv_names
                ]

        container = ttk.Frame(self.classes_tab)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        # left column: list of classes and controls
        left = ttk.Frame(container)
        left.grid(row=0, column=0, sticky="ns")
        left.rowconfigure(0, weight=1)

        self.cls_list = tk.Listbox(left, exportselection=False)
        self.cls_list.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.cls_list.bind("<<ListboxSelect>>", lambda _e: self._on_class_select())
        for name in sorted(self._classes_map.keys()):
            self.cls_list.insert(tk.END, name)

        left_btns = ttk.Frame(left)
        left_btns.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))
        ttk.Button(left_btns, text="Add", command=self._add_class).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_btns, text="Duplicate", command=self._dup_class).pack(side=tk.LEFT, padx=2)
        ttk.Button(left_btns, text="Delete", command=self._del_class).pack(side=tk.LEFT, padx=2)
        self.cls_save_btn = ttk.Button(left_btns, text="Save", command=self._save_classes, state="disabled")
        self.cls_save_btn.pack(side=tk.RIGHT, padx=2)

        # right column: form for class details
        right = ttk.Frame(container)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        form = ttk.Frame(right)
        form.grid(row=0, column=0, sticky="nsew")
        form.columnconfigure(1, weight=1)
        for r in (4, 5, 6, 8):
            form.rowconfigure(r, weight=1)

        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w")
        self.cls_name = tk.StringVar()
        name_entry = ttk.Entry(form, textvariable=self.cls_name)
        name_entry.grid(row=0, column=1, sticky="ew", padx=4)
        self.cls_name.trace_add("write", self._mark_classes_dirty)

        ttk.Label(form, text="Triggering PDV").grid(row=1, column=0, sticky="w")
        self.cls_trig = tk.StringVar()
        self.cls_trig_cb = ttk.Combobox(
            form, textvariable=self.cls_trig, values=self._pdv_names, state="readonly"
        )
        self.cls_trig_cb.grid(row=1, column=1, sticky="ew", padx=4)
        self.cls_trig_cb.bind("<<ComboboxSelected>>", lambda _e: self._mark_classes_dirty())

        ttk.Label(form, text="Model").grid(row=2, column=0, sticky="w")
        self.cls_model = tk.StringVar()
        models = ["<inherit global>"] + self._fetch_models()
        self.cls_model_cb = ttk.Combobox(form, textvariable=self.cls_model, values=models, state="readonly")
        model_row = ttk.Frame(form)
        model_row.grid(row=2, column=1, sticky="ew", padx=4)
        self.cls_model_cb.pack(in_=model_row, side=tk.LEFT, fill=tk.X, expand=True)
        self.cls_model_cb.bind("<<ComboboxSelected>>", lambda _e: self._mark_classes_dirty())
        ttk.Button(model_row, text="Refresh models", command=self._reload_model_choices).pack(side=tk.LEFT, padx=6)

        ttk.Label(form, text="Temperature").grid(row=3, column=0, sticky="w")
        self.cls_temp = tk.DoubleVar(value=1.0)
        self.cls_temp_inherit = tk.BooleanVar(value=False)
        temp_row = ttk.Frame(form)
        temp_row.grid(row=3, column=1, sticky="ew", padx=4)
        self.cls_temp_scale = ttk.Scale(temp_row, from_=0.0, to=2.0, orient=tk.HORIZONTAL, variable=self.cls_temp)
        self.cls_temp_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cls_temp_scale.bind("<ButtonPress-1>", self._enable_temp_override)
        self.cls_temp_label = ttk.Label(temp_row, text="1.00")
        self.cls_temp_label.pack(side=tk.LEFT, padx=6)
        ttk.Button(temp_row, text="Reset to inherited", command=self._temp_reset_inherit).pack(side=tk.LEFT, padx=6)
        self.cls_temp.trace_add("write", lambda *_: self._on_temp_changed())

        def _mk_label(row: int, text: str) -> None:
            ttk.Label(form, text=text).grid(row=row, column=0, sticky="nw")

        def _mk_st(row: int) -> scrolledtext.ScrolledText:
            st = scrolledtext.ScrolledText(form, height=4, wrap=tk.WORD)
            st.grid(row=row, column=1, sticky="nsew", padx=4, pady=2)
            return st

        _mk_label(4, "System Prompt")
        self.cls_sys = _mk_st(4)
        ttk.Button(form, text="Reset to inherited", command=lambda: self._reset_text(self.cls_sys)).grid(
            row=4, column=2, sticky="w"
        )
        _mk_label(5, "Pre Context")
        self.cls_pre = _mk_st(5)
        ttk.Button(form, text="Reset to inherited", command=lambda: self._reset_text(self.cls_pre)).grid(
            row=5, column=2, sticky="w"
        )
        _mk_label(6, "Post Context")
        self.cls_post = _mk_st(6)
        ttk.Button(form, text="Reset to inherited", command=lambda: self._reset_text(self.cls_post)).grid(
            row=6, column=2, sticky="w"
        )

        ttk.Label(form, text="Flags").grid(row=7, column=0, sticky="nw")
        flags = ttk.Frame(form)
        flags.grid(row=7, column=1, sticky="w", padx=4, pady=2)
        self.cls_readq = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            flags,
            text="Reads message queue",
            variable=self.cls_readq,
            command=lambda: self._mark_classes_dirty(),
        ).pack(side=tk.LEFT, padx=(0, 6))
        self.cls_outdisc = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            flags,
            text="Outputs to Discord",
            variable=self.cls_outdisc,
            command=lambda: self._mark_classes_dirty(),
        ).pack(side=tk.LEFT, padx=(0, 6))
        self.cls_arch = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            flags,
            text="Archivist",
            variable=self.cls_arch,
            command=lambda: self._mark_classes_dirty(),
        ).pack(side=tk.LEFT, padx=(0, 6))
        # Per-class switches for excluding global contexts
        self.cls_ign_sys = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            flags,
            text="Ignore Global System",
            variable=self.cls_ign_sys,
            command=lambda: self._mark_classes_dirty(),
        ).pack(side=tk.LEFT, padx=(12, 6))
        self.cls_ign_pre = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            flags,
            text="Ignore Global Pre",
            variable=self.cls_ign_pre,
            command=lambda: self._mark_classes_dirty(),
        ).pack(side=tk.LEFT, padx=(0, 6))
        self.cls_ign_post = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            flags,
            text="Ignore Global Post",
            variable=self.cls_ign_post,
            command=lambda: self._mark_classes_dirty(),
        ).pack(side=tk.LEFT)

        ttk.Label(form, text="PDV Adjustments").grid(row=8, column=0, sticky="nw")
        scf = ttk.Frame(form)
        scf.grid(row=8, column=1, sticky="nsew", padx=4, pady=2)
        canvas = tk.Canvas(scf, highlightthickness=0)
        vsb = ttk.Scrollbar(scf, orient="vertical", command=canvas.yview)
        self.pdv_rows_container = ttk.Frame(canvas)
        self.pdv_rows_container.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        window_id = canvas.create_window((0, 0), window=self.pdv_rows_container, anchor="nw")

        def _resize(_event) -> None:
            canvas.itemconfigure(window_id, width=canvas.winfo_width())

        canvas.bind("<Configure>", _resize)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self._pdv_row_widgets: list[tuple[ttk.Combobox, tk.DoubleVar, ttk.Scale, ttk.Button]] = []
        ttk.Button(
            form,
            text="Add PDV Adjustment",
            command=lambda: self._pdv_adj_add_row(None, 0.0),
        ).grid(row=9, column=1, sticky="w", padx=4, pady=(0, 4))

        self._clear_class_form()
        self._refresh_agent_class_choices()

    def _clear_class_form(self) -> None:
        self._loading_class = True
        try:
            self.cls_name.set("")
            default_trig = self._pdv_names[0] if self._pdv_names else ""
            self.cls_trig.set(default_trig)
            self.cls_model.set("<inherit global>")
            self.cls_temp.set(1.0)
            self.cls_temp_inherit.set(True)
            self._clear_pdv_rows()
            for widget in (self.cls_sys, self.cls_pre, self.cls_post):
                widget.delete("1.0", tk.END)
            self.cls_readq.set(False)
            self.cls_outdisc.set(False)
            self.cls_arch.set(False)
            self.cls_ign_sys.set(False)
            self.cls_ign_pre.set(False)
            self.cls_ign_post.set(False)
        finally:
            self._loading_class = False
        self._temp_apply_inherit_state()
        self._set_classes_dirty(False)

    def _clear_pdv_rows(self) -> None:
        for cb, var, sc, rm in list(getattr(self, "_pdv_row_widgets", [])):
            try:
                row = cb.master
            except Exception:
                row = None
            if row is not None and row.winfo_exists():
                row.destroy()
        self._pdv_row_widgets.clear()

    def _mark_classes_dirty(self, *_args, **_kwargs) -> None:
        if self._loading_class:
            return
        self._set_classes_dirty(True)

    def _mark_classes_dirty_from_text(self, _event=None) -> None:
        self._mark_classes_dirty()

    def _on_class_select(self) -> None:
        # warn if dirty before navigating
        if not self._confirm_discard_if_dirty():
            return
        sel = self.cls_list.curselection()
        if not sel:
            self._clear_class_form()
            return
        name = self.cls_list.get(sel[0])
        c = self._classes_map.get(name, {})
        self._loading_class = True
        try:
            self.cls_name.set(c.get("name", name))
            trig = c.get("triggering_pdv", "")
            if trig and trig not in self._pdv_names:
                trig = ""
            if not trig and self._pdv_names:
                trig = self._pdv_names[0]
            self.cls_trig.set(trig)
            model = c.get("model") or "<inherit global>"
            values = list(self.cls_model_cb["values"])
            if model not in values:
                values.append(model)
                self.cls_model_cb.configure(values=values)
            self.cls_model.set(model)
            t = c.get("temperature", None)
            if t is None:
                self.cls_temp_inherit.set(True)
                self.cls_temp.set(1.0)
            else:
                self.cls_temp_inherit.set(False)
                try:
                    self.cls_temp.set(float(t))
                except Exception:
                    self.cls_temp.set(1.0)
            for widget, key in (
                (self.cls_sys, "system_prompt"),
                (self.cls_pre, "pre_context_message"),
                (self.cls_post, "post_context_message"),
            ):
                widget.delete("1.0", tk.END)
                val = c.get(key)
                if val:
                    widget.insert("1.0", val)
            self.cls_readq.set(bool(c.get("reads_message_queue")))
            self.cls_outdisc.set(bool(c.get("outputs_to_discord")))
            self.cls_arch.set(bool(c.get("is_archivist")))
            self.cls_ign_sys.set(bool(c.get("ignore_global_system", False)))
            self.cls_ign_pre.set(bool(c.get("ignore_global_pre", False)))
            self.cls_ign_post.set(bool(c.get("ignore_global_post", False)))
            self._clear_pdv_rows()
            for adj in c.get("pdv_adjustments", []):
                if adj.get("name") in self._pdv_names:
                    try:
                        delta = float(adj.get("delta", adj.get("delta_pct", 0.0)))
                    except Exception:
                        delta = 0.0
                    self._pdv_adj_add_row(adj.get("name"), delta)
        finally:
            self._loading_class = False
        self._temp_apply_inherit_state()
        self._set_classes_dirty(False)
        self._save_ui_state({"last_class": name})

    def _add_class(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        base = "NewClass"
        idx = 1
        name = base
        while name in self._classes_map:
            idx += 1
            name = f"{base}{idx}"
        new_cls = {
            "name": name,
            "triggering_pdv": self._pdv_names[0] if self._pdv_names else "",
            "reads_message_queue": False,
            "outputs_to_discord": False,
            "is_archivist": False,
            "ignore_global_system": False,
            "ignore_global_pre": False,
            "ignore_global_post": False,
        }
        self._classes_map[name] = new_cls
        self.cls_list.insert(tk.END, name)
        self.cls_list.selection_clear(0, tk.END)
        self.cls_list.selection_set(tk.END)
        self._on_class_select()
        self._set_classes_dirty(True)
        self._refresh_agent_class_choices()

    def _dup_class(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        sel = self.cls_list.curselection()
        if not sel:
            return
        src = self.cls_list.get(sel[0])
        c = json.loads(json.dumps(self._classes_map.get(src, {})))
        base = f"{src}_copy"
        i = 1
        name = base
        while name in self._classes_map:
            i += 1
            name = f"{base}{i}"
        c["name"] = name
        self._classes_map[name] = c
        self.cls_list.insert(tk.END, name)
        self.cls_list.selection_clear(0, tk.END)
        self.cls_list.selection_set(tk.END)
        self._on_class_select()
        self._set_classes_dirty(True)
        self._refresh_agent_class_choices()

    def _del_class(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        sel = self.cls_list.curselection()
        if not sel:
            return
        name = self.cls_list.get(sel[0])
        if not messagebox.askyesno("Delete Class", f"Delete agent class '{name}'?"):
            return
        self.cls_list.delete(sel[0])
        self._classes_map.pop(name, None)
        if self.cls_list.size() > 0:
            new_idx = min(sel[0], self.cls_list.size() - 1)
            self.cls_list.selection_set(new_idx)
            self._on_class_select()
        else:
            self._clear_class_form()
        self._set_classes_dirty(True)
        self._refresh_agent_class_choices()

    def _pdv_adj_add_row(self, name: Optional[str], delta: float) -> None:
        existing = {cb.get() for cb, *_ in self._pdv_row_widgets if cb.winfo_exists()}
        choices = [p for p in self._pdv_names if p not in existing or p == name]
        if not choices:
            messagebox.showinfo("PDVs", "All PDVs already added.")
            return
        row = ttk.Frame(self.pdv_rows_container)
        row.pack(fill=tk.X, pady=2)
        cb = ttk.Combobox(row, values=choices, state="readonly")
        cb.set(name or choices[0])
        cb.pack(side=tk.LEFT, padx=4)
        cb.bind(
            "<<ComboboxSelected>>",
            lambda _e: (self._refresh_class_pdv_choices(), self._mark_classes_dirty()),
        )
        val = float(delta) if isinstance(delta, (int, float)) else 0.0
        val = max(-1.0, min(1.0, val))
        var = tk.DoubleVar(value=val)
        sc = ttk.Scale(row, from_=-1.0, to=1.0, orient=tk.HORIZONTAL, variable=var)
        sc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        lbl = ttk.Label(row, text=f"{var.get():+0.2f}")
        lbl.pack(side=tk.LEFT, padx=4)

        def _on_var_change(*_args) -> None:
            lbl.config(text=f"{var.get():+0.2f}")
            self._mark_classes_dirty()

        var.trace_add("write", _on_var_change)
        rm = ttk.Button(row, text="Remove")

        def _remove_row() -> None:
            if row.winfo_exists():
                row.destroy()
            try:
                self._pdv_row_widgets.remove((cb, var, sc, rm))
            except ValueError:
                pass
            self._mark_classes_dirty()

        rm.configure(command=_remove_row)
        rm.pack(side=tk.LEFT, padx=4)
        self._pdv_row_widgets.append((cb, var, sc, rm))
        self._refresh_class_pdv_choices()
        self._mark_classes_dirty()

    def _collect_form(self) -> dict:
        c = {
            "name": self.cls_name.get().strip(),
            "triggering_pdv": self.cls_trig.get().strip(),
            "model": (None if self.cls_model.get() == "<inherit global>" else (self.cls_model.get().strip() or None)),
            "temperature": (
                None
                if self.cls_temp_inherit.get()
                else round(min(max(float(self.cls_temp.get()), 0.0), 2.0), 2)
            ),
            "reads_message_queue": bool(self.cls_readq.get()),
            "outputs_to_discord": bool(self.cls_outdisc.get()),
            "is_archivist": bool(self.cls_arch.get()),
            "ignore_global_system": bool(self.cls_ign_sys.get()),
            "ignore_global_pre": bool(self.cls_ign_pre.get()),
            "ignore_global_post": bool(self.cls_ign_post.get()),
        }
        sys_txt = self.cls_sys.get("1.0", tk.END).strip()
        pre_txt = self.cls_pre.get("1.0", tk.END).strip()
        post_txt = self.cls_post.get("1.0", tk.END).strip()
        if sys_txt:
            c["system_prompt"] = sys_txt
        if pre_txt:
            c["pre_context_message"] = pre_txt
        if post_txt:
            c["post_context_message"] = post_txt
        adjs: list[dict[str, float]] = []
        for cb, var, _, _ in self._pdv_row_widgets:
            if not cb.winfo_exists():
                continue
            n = cb.get().strip()
            if not n:
                continue
            adjs.append({"name": n, "delta": float(max(-1.0, min(1.0, var.get())))})
        if adjs:
            c["pdv_adjustments"] = adjs
        return c

    def _apply_form_to_current(self) -> Optional[str]:
        sel = self.cls_list.curselection()
        if not sel:
            return None
        idx = sel[0]
        old_name = self.cls_list.get(idx)
        c = self._collect_form()
        name = c.get("name") or old_name
        if not name:
            messagebox.showerror("Classes", "Name is required.")
            return None
        if not c.get("triggering_pdv"):
            messagebox.showerror("Classes", "Triggering PDV is required.")
            return None
        if name != old_name and name in self._classes_map:
            messagebox.showerror("Classes", f"Name '{name}' already exists.")
            return None
        self._classes_map.pop(old_name, None)
        self._classes_map[name] = c
        self.cls_list.delete(idx)
        self.cls_list.insert(idx, name)
        self.cls_list.selection_set(idx)
        self._set_classes_dirty(True)
        return name

    def _refresh_class_pdv_choices(self) -> None:
        if hasattr(self, "cls_trig_cb"):
            self.cls_trig_cb.configure(values=self._pdv_names)
            if self.cls_trig.get() not in self._pdv_names:
                self.cls_trig.set(self._pdv_names[0] if self._pdv_names else "")
        removed = False
        for cb, var, sc, rm in list(self._pdv_row_widgets):
            if not cb.winfo_exists():
                continue
            current = cb.get()
            others = {
                other_cb.get()
                for other_cb, *_ in self._pdv_row_widgets
                if other_cb.winfo_exists() and other_cb is not cb
            }
            choices = [p for p in self._pdv_names if p not in others or p == current]
            cb.configure(values=choices)
            if current not in choices:
                if choices:
                    cb.set(choices[0])
                else:
                    parent = cb.master
                    if parent.winfo_exists():
                        parent.destroy()
                    try:
                        self._pdv_row_widgets.remove((cb, var, sc, rm))
                    except ValueError:
                        pass
                    removed = True
        if removed and not self._loading_class:
            self._mark_classes_dirty()

    def _save_classes(self) -> None:
        cur = self._apply_form_to_current()
        if cur is None:
            return
        seen = set()
        for cname, c in self._classes_map.items():
            n = (c.get("name") or "").strip()
            tpdv = (c.get("triggering_pdv") or "").strip()
            if not n or not tpdv:
                messagebox.showerror("Agent Classes", f"Class '{cname}' is missing required fields.")
                return
            if n in seen:
                messagebox.showerror("Agent Classes", f"Duplicate class name '{n}'.")
                return
            seen.add(n)
            c["name"] = n
            if not c.get("model"):
                c["model"] = None
            if "temperature" in c and c["temperature"] is None:
                pass
            elif "temperature" in c:
                try:
                    c["temperature"] = round(min(max(float(c["temperature"]), 0.0), 2.0), 2)
                except Exception:
                    c["temperature"] = None
            if "pdv_adjustments" in c:
                c["pdv_adjustments"] = [
                    {"name": a.get("name"),
                     "delta": max(-1.0, min(1.0, float(a.get("delta", 0.0))))}
                    for a in c.get("pdv_adjustments", [])
                    if a.get("name") in self._pdv_names
                ]
        save_classes(self._classes_map)
        self._update_required_configs_state()
        self._set_classes_dirty(False)
        messagebox.showinfo("Agent Classes", "Saved.")
        self._save_ui_state({"last_class": cur})
        self._refresh_agent_class_choices()

    # ----- helpers for classes editor -----
    def _set_classes_dirty(self, dirty: bool) -> None:
        self._classes_dirty = bool(dirty)
        try:
            if self._classes_dirty:
                self.cls_save_btn.state(["!disabled"])
            else:
                self.cls_save_btn.state(["disabled"])
        except Exception:
            pass

    def _confirm_discard_if_dirty(self) -> bool:
        if not getattr(self, "_classes_dirty", False):
            return True
        res = messagebox.askyesno("Unsaved changes", "Save changes to Agent Classes before leaving?")
        if res:
            self._save_classes()
            return not self._classes_dirty
        self._set_classes_dirty(False)
        return True

    def _on_config_tab_changed(self, _e=None) -> None:
        selected = self.config_nb.select()
        if str(selected) != str(self.classes_tab):
            self._confirm_discard_if_dirty()

    def _on_temp_changed(self) -> None:
        try:
            val = float(self.cls_temp.get())
        except Exception:
            val = 0.0
        if self.cls_temp_inherit.get():
            self.cls_temp_label.config(text="(inherit)")
        else:
            self.cls_temp_label.config(text=f"{val:.2f}")
            if not self._loading_class:
                self._set_classes_dirty(True)

    def _temp_reset_inherit(self) -> None:
        self.cls_temp_inherit.set(True)
        self._temp_apply_inherit_state()
        if not self._loading_class:
            self._set_classes_dirty(True)

    def _temp_apply_inherit_state(self) -> None:
        if self.cls_temp_inherit.get():
            self.cls_temp_scale.state(["disabled"])
            self.cls_temp_label.config(text="(inherit)")
        else:
            self.cls_temp_scale.state(["!disabled"])
            try:
                val = float(self.cls_temp.get())
            except Exception:
                val = 0.0
            self.cls_temp_label.config(text=f"{val:.2f}")

    def _enable_temp_override(self, *_event) -> None:
        if self.cls_temp_inherit.get():
            self.cls_temp_inherit.set(False)
            self._temp_apply_inherit_state()
            if not self._loading_class:
                self._set_classes_dirty(True)

    def _reset_text(self, st: scrolledtext.ScrolledText) -> None:
        st.delete("1.0", tk.END)
        if not self._loading_class:
            self._set_classes_dirty(True)

    def _reload_model_choices(self) -> None:
        try:
            self.cls_model_cb["values"] = ["<inherit global>"] + self._fetch_models()
        except Exception:
            pass
        if not self._loading_class:
            self._set_classes_dirty(True)

    def _save_ui_state(self, data: dict) -> None:
        try:
            os.makedirs(CHATLOG_DIR, exist_ok=True)
            path = os.path.join(CHATLOG_DIR, "ui_state.json")
            cur = {}
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    cur = json.load(f) or {}
            cur.update(data or {})
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cur, f)
        except Exception:
            pass

    def _load_ui_state(self) -> dict:
        try:
            path = os.path.join(CHATLOG_DIR, "ui_state.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            pass
        return {}

    def _build_agents_editor_tab(self) -> None:
        frame = getattr(self, "agents_editor_tab", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        self._agents_model = self._ui_agents()

        paned = ttk.Panedwindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        self.agent_list = tk.Listbox(left, exportselection=False)
        self.agent_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 2))
        self.agent_list.bind("<<ListboxSelect>>", self._agent_load_into_form)

        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(btns, text="Add", command=self._agent_add).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Duplicate", command=self._agent_dup).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Delete", command=self._agent_del).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Save", command=self._agent_save_all).pack(side=tk.RIGHT, padx=2)

        form = ttk.Frame(right, padding=(8, 8))
        form.pack(fill=tk.BOTH, expand=True)
        form.columnconfigure(1, weight=1)

        self._agent_form_vars = {
            "name": tk.StringVar(),
            "agent_class": tk.StringVar(),
            "model": tk.StringVar(),
            "temperature": tk.StringVar(),
            "topic_prompt": tk.StringVar(),
            "role_prompt": tk.StringVar(),
            "watchdog_timeout": tk.StringVar(),
            "max_tokens": tk.StringVar(),
        }
        self._agent_text_fields = {}

        row = 0
        ttk.Label(form, text="Name").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(form, textvariable=self._agent_form_vars["name"]).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1

        ttk.Label(form, text="Agent Class").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        self._agent_class_cb = ttk.Combobox(
            form,
            textvariable=self._agent_form_vars["agent_class"],
            values=sorted(self._classes_map.keys()),
        )
        self._agent_class_cb.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(form, text="Model").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(form, textvariable=self._agent_form_vars["model"]).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1

        self._agent_model_hint = ttk.Label(form, text="", foreground="#666666")
        self._agent_model_hint.grid(row=row, column=1, sticky="w", pady=(0, 4))
        row += 1

        ttk.Label(form, text="Temperature").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(form, textvariable=self._agent_form_vars["temperature"]).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1

        ttk.Label(form, text="Topic Prompt").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(form, textvariable=self._agent_form_vars["topic_prompt"]).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1

        ttk.Label(form, text="Role Prompt").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(form, textvariable=self._agent_form_vars["role_prompt"]).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1

        ttk.Label(form, text="Watchdog Timeout").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(form, textvariable=self._agent_form_vars["watchdog_timeout"]).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1

        ttk.Label(form, text="Max Tokens").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(form, textvariable=self._agent_form_vars["max_tokens"]).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1

        def _add_text_field(label: str, key: str, height: int = 5) -> None:
            nonlocal row
            ttk.Label(form, text=label).grid(
                row=row, column=0, sticky="nw", padx=(0, 8), pady=(6, 2)
            )
            txt = scrolledtext.ScrolledText(form, height=height, wrap=tk.WORD)
            txt.grid(row=row, column=1, sticky="nsew", pady=(6, 2))
            form.rowconfigure(row, weight=1)
            self._agent_text_fields[key] = txt
            row += 1

        _add_text_field("System Prompt", "system_prompt", height=6)
        _add_text_field("Pre-context Message", "pre_context_message", height=4)
        _add_text_field("Post-context Message", "post_context_message", height=4)
        _add_text_field("Chat Style", "chat_style", height=4)

        ttk.Label(form, text="Flags").grid(
            row=row, column=0, sticky="nw", padx=(0, 8), pady=(6, 2)
        )
        flags = ttk.Frame(form)
        flags.grid(row=row, column=1, sticky="w", pady=(6, 2))
        row += 1

        self.ag_ign_glob_sys = tk.BooleanVar(value=False)
        self.ag_ign_glob_pre = tk.BooleanVar(value=False)
        self.ag_ign_glob_post = tk.BooleanVar(value=False)
        self.ag_ign_cls_sys = tk.BooleanVar(value=False)
        self.ag_ign_cls_pre = tk.BooleanVar(value=False)
        self.ag_ign_cls_post = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            flags,
            text="Ignore Global System",
            variable=self.ag_ign_glob_sys,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Checkbutton(
            flags,
            text="Ignore Global Pre",
            variable=self.ag_ign_glob_pre,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Checkbutton(
            flags,
            text="Ignore Global Post",
            variable=self.ag_ign_glob_post,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(
            flags,
            text="Ignore Class System",
            variable=self.ag_ign_cls_sys,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Checkbutton(
            flags,
            text="Ignore Class Pre",
            variable=self.ag_ign_cls_pre,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Checkbutton(
            flags,
            text="Ignore Class Post",
            variable=self.ag_ign_cls_post,
        ).pack(side=tk.LEFT)

        self._active_agent_index = None
        self._loading_agent_form = False
        self._agent_form_vars["model"].trace_add("write", self._update_agent_model_hint)
        self._update_agent_model_hint()
        self._clear_agent_form()
        self._refresh_agent_listbox()

    def _selected_agent_index(self) -> Optional[int]:
        if self.agent_list is None:
            return None
        selection = self.agent_list.curselection()
        if not selection:
            return None
        return int(selection[0])

    def _agent_load_into_form(self, *_event) -> None:
        if self.agent_list is None:
            return
        idx = self._selected_agent_index()
        if idx is None or idx >= len(self._agents_model):
            self._active_agent_index = None
            self._clear_agent_form()
            return
        if idx != self._active_agent_index:
            self._agent_store_from_form()
        agent = self._agents_model[idx]
        self._active_agent_index = idx
        self._loading_agent_form = True
        name_var = self._agent_form_vars.get("name")
        if name_var is not None:
            name_var.set(str(agent.get("name", "")))
        cls_var = self._agent_form_vars.get("agent_class")
        if cls_var is not None:
            cls_var.set(str(agent.get("agent_class", "")))
        model_var = self._agent_form_vars.get("model")
        if model_var is not None:
            model_val = agent.get("model")
            model_var.set("" if model_val is None else str(model_val))
        temp_var = self._agent_form_vars.get("temperature")
        if temp_var is not None:
            temp_val = agent.get("temperature")
            temp_var.set("" if temp_val in (None, "") else str(temp_val))
        topic_var = self._agent_form_vars.get("topic_prompt")
        if topic_var is not None:
            topic_var.set(str(agent.get("topic_prompt", "")))
        role_var = self._agent_form_vars.get("role_prompt")
        if role_var is not None:
            role_var.set(str(agent.get("role_prompt", "")))
        wd_var = self._agent_form_vars.get("watchdog_timeout")
        if wd_var is not None:
            wd_val = agent.get("watchdog_timeout")
            wd_var.set("" if wd_val in (None, "") else str(wd_val))
        max_var = self._agent_form_vars.get("max_tokens")
        if max_var is not None:
            max_val = agent.get("max_tokens")
            max_var.set("" if max_val in (None, "") else str(max_val))
        for key, widget in self._agent_text_fields.items():
            widget.delete("1.0", tk.END)
            text_val = agent.get(key)
            if text_val:
                widget.insert(tk.END, text_val)
        if hasattr(self, "ag_ign_glob_sys"):
            self.ag_ign_glob_sys.set(bool(agent.get("ignore_global_system", False)))
            self.ag_ign_glob_pre.set(bool(agent.get("ignore_global_pre", False)))
            self.ag_ign_glob_post.set(bool(agent.get("ignore_global_post", False)))
            self.ag_ign_cls_sys.set(bool(agent.get("ignore_class_system", False)))
            self.ag_ign_cls_pre.set(bool(agent.get("ignore_class_pre", False)))
            self.ag_ign_cls_post.set(bool(agent.get("ignore_class_post", False)))
        self._loading_agent_form = False
        self._update_agent_model_hint()

    def _clear_agent_form(self) -> None:
        if not self._agent_form_vars:
            return
        self._loading_agent_form = True
        for var in self._agent_form_vars.values():
            var.set("")
        for widget in self._agent_text_fields.values():
            widget.delete("1.0", tk.END)
        if hasattr(self, "ag_ign_glob_sys"):
            self.ag_ign_glob_sys.set(False)
            self.ag_ign_glob_pre.set(False)
            self.ag_ign_glob_post.set(False)
            self.ag_ign_cls_sys.set(False)
            self.ag_ign_cls_pre.set(False)
            self.ag_ign_cls_post.set(False)
        self._loading_agent_form = False
        self._update_agent_model_hint()

    def _agent_store_from_form(self) -> None:
        if (
            self._active_agent_index is None
            or self._active_agent_index >= len(self._agents_model)
            or not self._agent_form_vars
        ):
            return
        agent = self._agents_model[self._active_agent_index]

        name_var = self._agent_form_vars.get("name")
        if name_var is not None:
            agent["name"] = name_var.get().strip()

        cls_var = self._agent_form_vars.get("agent_class")
        if cls_var is not None:
            agent["agent_class"] = cls_var.get().strip()

        def _assign_optional_str(field: str) -> None:
            var = self._agent_form_vars.get(field)
            if var is None:
                return
            value = var.get().strip()
            if value:
                agent[field] = value
            else:
                agent.pop(field, None)

        for optional in ("model", "topic_prompt", "role_prompt"):
            _assign_optional_str(optional)

        def _assign_optional_int(field: str) -> None:
            var = self._agent_form_vars.get(field)
            if var is None:
                return
            value = var.get().strip()
            if not value:
                agent.pop(field, None)
                return
            try:
                agent[field] = int(value)
            except ValueError:
                agent[field] = value

        for numeric in ("watchdog_timeout", "max_tokens"):
            _assign_optional_int(numeric)

        temp_var = self._agent_form_vars.get("temperature")
        if temp_var is not None:
            value = temp_var.get().strip()
            if not value:
                agent.pop("temperature", None)
            else:
                try:
                    agent["temperature"] = float(value)
                except ValueError:
                    agent["temperature"] = value

        for key, widget in self._agent_text_fields.items():
            value = widget.get("1.0", tk.END).strip()
            if value:
                agent[key] = value
            else:
                agent.pop(key, None)

        if hasattr(self, "ag_ign_glob_sys"):
            agent["ignore_global_system"] = bool(self.ag_ign_glob_sys.get())
            agent["ignore_global_pre"] = bool(self.ag_ign_glob_pre.get())
            agent["ignore_global_post"] = bool(self.ag_ign_glob_post.get())
            agent["ignore_class_system"] = bool(self.ag_ign_cls_sys.get())
            agent["ignore_class_pre"] = bool(self.ag_ign_cls_pre.get())
            agent["ignore_class_post"] = bool(self.ag_ign_cls_post.get())

        agent.setdefault("groups_in", [])
        agent.setdefault("groups_out", [])

    def _refresh_agent_listbox(self, select_index: Optional[int] = None) -> None:
        if self.agent_list is None:
            return
        self._agent_store_from_form()
        self.agent_list.delete(0, tk.END)
        for agent in self._agents_model:
            label = agent.get("name") or "<unnamed>"
            self.agent_list.insert(tk.END, label)
        if not self._agents_model:
            self.agent_list.selection_clear(0, tk.END)
            self._active_agent_index = None
            self._clear_agent_form()
            return
        if select_index is None:
            if self._active_agent_index is not None and self._active_agent_index < len(
                self._agents_model
            ):
                select_index = self._active_agent_index
            else:
                select_index = 0
        select_index = max(0, min(select_index, len(self._agents_model) - 1))
        self.agent_list.selection_clear(0, tk.END)
        self.agent_list.selection_set(select_index)
        self.agent_list.see(select_index)
        self._agent_load_into_form()

    def _agent_add(self) -> None:
        self._agent_store_from_form()
        base_name = f"Agent{len(self._agents_model) + 1}"
        existing = {a.get("name") for a in self._agents_model if a.get("name")}
        candidate = base_name
        suffix = 1
        while candidate in existing:
            suffix += 1
            candidate = f"{base_name}_{suffix}"
        new_agent = {
            "name": candidate,
            "agent_class": "",
            "groups_in": [],
            "groups_out": [],
            "ignore_global_system": False,
            "ignore_global_pre": False,
            "ignore_global_post": False,
            "ignore_class_system": False,
            "ignore_class_pre": False,
            "ignore_class_post": False,
        }
        self._agents_model.append(new_agent)
        self._refresh_agent_listbox(select_index=len(self._agents_model) - 1)

    def _agent_dup(self) -> None:
        idx = self._selected_agent_index()
        if idx is None or idx >= len(self._agents_model):
            return
        self._agent_store_from_form()
        dup = json.loads(json.dumps(self._agents_model[idx]))
        base = dup.get("name") or "Agent"
        existing = {a.get("name") for a in self._agents_model if a.get("name")}
        candidate = f"{base}_copy"
        suffix = 1
        while candidate in existing:
            suffix += 1
            candidate = f"{base}_copy{suffix}"
        dup["name"] = candidate
        dup.setdefault("groups_in", [])
        dup.setdefault("groups_out", [])
        self._agents_model.append(dup)
        self._refresh_agent_listbox(select_index=len(self._agents_model) - 1)

    def _agent_del(self) -> None:
        idx = self._selected_agent_index()
        if idx is None or idx >= len(self._agents_model):
            return
        self._agent_store_from_form()
        del self._agents_model[idx]
        if not self._agents_model:
            self._refresh_agent_listbox()
        else:
            self._refresh_agent_listbox(select_index=max(0, idx - 1))

    def _agent_save_all(self) -> None:
        self._agent_store_from_form()
        cleaned: list[dict] = []
        for idx, agent in enumerate(self._agents_model):
            name = agent.get("name", "").strip()
            if not name:
                messagebox.showwarning("Agents", f"Agent #{idx + 1} must have a name before saving.")
                return
            agent_class = agent.get("agent_class", "").strip()
            if not agent_class:
                messagebox.showwarning(
                    "Agents", f"Agent '{name}' must have an agent_class before saving."
                )
                return
            temp_val = agent.get("temperature")
            if isinstance(temp_val, str) and temp_val:
                messagebox.showwarning(
                    "Agents", f"Temperature for '{name}' must be a number (got '{temp_val}')."
                )
                return
            for field in ("watchdog_timeout", "max_tokens"):
                raw_val = agent.get(field)
                if isinstance(raw_val, str) and raw_val:
                    messagebox.showwarning(
                        "Agents",
                        f"{field.replace('_', ' ').title()} for '{name}' must be an integer (got '{raw_val}').",
                    )
                    return
            copy = json.loads(json.dumps(agent))
            copy.setdefault("groups_in", [])
            copy.setdefault("groups_out", [])
            cleaned.append(copy)
        try:
            save_agents(cleaned)
        except Exception as exc:
            messagebox.showerror("Agents", f"Failed to save agents: {exc}")
            return
        self._agents_model = self._ui_agents()
        self._update_required_configs_state()
        messagebox.showinfo("Agents", "Saved agents.json")
        self._refresh_agent_listbox()
        self._build_simple_groups_tab()

    def _update_agent_model_hint(self, *_args) -> None:
        if self._agent_model_hint is None:
            return
        model_var = self._agent_form_vars.get("model") if self._agent_form_vars else None
        if model_var is None:
            self._agent_model_hint.config(text="")
            return
        text = "Use class/global default" if not model_var.get().strip() else ""
        self._agent_model_hint.config(text=text)

    def _refresh_agent_class_choices(self) -> None:
        if self._agent_class_cb is None:
            return
        try:
            self._agent_class_cb["values"] = sorted(self._classes_map.keys())
        except Exception:
            pass

    def _build_simple_groups_tab(self) -> None:
        frame = getattr(self, "simple_groups_tab", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        agents = self._ui_agents()
        self._agents_cache = agents

        flagged = [a for a in agents if a.get("flag_no_downstream")]
        if flagged:
            banner = ttk.Frame(frame, relief=tk.RIDGE, borderwidth=1)
            banner.pack(fill=tk.X, pady=2)
            btop = ttk.Frame(banner)
            btop.pack(fill=tk.X)
            ttk.Label(btop, text="Agents with no downstream:", foreground="red").pack(side=tk.LEFT)
            ttk.Button(btop, text="Dismiss", command=banner.destroy).pack(side=tk.RIGHT)
            self._flag_list = tk.Listbox(banner, height=min(5, len(flagged)), exportselection=False)
            for a in flagged:
                self._flag_list.insert(tk.END, a["name"])
            self._flag_list.pack(fill=tk.X)
            self._flag_list.bind("<<ListboxSelect>>", self._on_flag_select)
            self._flag_info = tk.Label(banner, justify="left")
            self._flag_info.pack(fill=tk.X, padx=4, pady=2)
            btns = ttk.Frame(banner)
            btns.pack(fill=tk.X, pady=4)
            ttk.Button(btns, text="Batch Wiring", command=self._batch_wiring).pack(side=tk.LEFT, padx=2)
            ttk.Button(btns, text="Repair Dead-Ends", command=self._repair_dead_ends).pack(side=tk.LEFT, padx=2)

        graph_container = ttk.Frame(frame)
        graph_container.pack(fill=tk.BOTH, expand=True)
        self._build_agents_groups_graph(graph_container, agents)

    # ---------- Agents ↔ Groups graph helpers ----------
    def _build_agents_groups_graph(self, parent: tk.Widget, agents: list[dict]) -> None:
        prev_selection = getattr(self, "_aggr_current_value", "")
        self._aggr_agents_by_name = {
            a.get("name", ""): a for a in agents if a.get("name")
        }
        all_groups = sorted(
            set(g for a in agents for g in (a.get("groups_in", []) or []))
            | set(g for a in agents for g in (a.get("groups_out", []) or []))
        )
        agent_names = sorted(self._aggr_agents_by_name)
        self._aggr_groups_senders: dict[str, list[str]] = {}
        self._aggr_groups_listeners: dict[str, list[str]] = {}
        for g in all_groups:
            self._aggr_groups_senders[g] = sorted(
                a.get("name", "")
                for a in agents
                if g in (a.get("groups_out", []) or []) and a.get("name")
            )
            self._aggr_groups_listeners[g] = sorted(
                a.get("name", "")
                for a in agents
                if g in (a.get("groups_in", []) or []) and a.get("name")
            )

        graph_frame = ttk.LabelFrame(parent, text="Agents ↔ Groups")
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        paned = ttk.Panedwindow(graph_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_pane = ttk.Frame(paned)
        paned.add(left_pane, weight=1)

        center_pane = ttk.Frame(paned)
        paned.add(center_pane, weight=3)

        right_pane = ttk.Frame(paned)
        paned.add(right_pane, weight=1)

        self._aggr_left_label = ttk.Label(left_pane, text="Left")
        self._aggr_left_label.pack(anchor="w", padx=4, pady=(4, 0))
        self._aggr_left_list = tk.Listbox(left_pane, exportselection=False)
        self._aggr_left_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self._aggr_left_list.bind("<Double-1>", lambda _e: self._aggr_add_left())
        self._aggr_left_add = ttk.Button(
            left_pane, text="Add", command=self._aggr_add_left, state="disabled"
        )
        self._aggr_left_add.pack(fill=tk.X, padx=4, pady=(0, 6))
        self._aggr_left_list.bind(
            "<<ListboxSelect>>", lambda _e: self._aggr_update_add_state("left")
        )

        top = ttk.Frame(center_pane)
        top.pack(fill=tk.X, padx=4, pady=(4, 2))
        ttk.Label(top, text="Select:").pack(side=tk.LEFT)
        values = [f"Group: {g}" for g in all_groups] + [
            f"Agent: {n}" for n in agent_names
        ]
        self._aggr_selector = ttk.Combobox(
            top,
            state="readonly",
            values=values,
            width=60,
        )
        self._aggr_selector.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._aggr_selector.bind("<<ComboboxSelected>>", self._aggr_on_select)

        self._aggr_canvas = tk.Canvas(center_pane, background="white", height=260)
        self._aggr_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 6))
        self._aggr_canvas.bind("<Configure>", lambda _e: self._aggr_redraw())

        self._aggr_right_label = ttk.Label(right_pane, text="Right")
        self._aggr_right_label.pack(anchor="w", padx=4, pady=(4, 0))
        self._aggr_right_list = tk.Listbox(right_pane, exportselection=False)
        self._aggr_right_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self._aggr_right_list.bind("<Double-1>", lambda _e: self._aggr_add_right())
        self._aggr_right_add = ttk.Button(
            right_pane, text="Add", command=self._aggr_add_right, state="disabled"
        )
        self._aggr_right_add.pack(fill=tk.X, padx=4, pady=(0, 6))
        self._aggr_right_list.bind(
            "<<ListboxSelect>>", lambda _e: self._aggr_update_add_state("right")
        )

        self._aggr_mode = None
        self._aggr_active_agent = None
        self._aggr_active_group = None
        self._aggr_node_items = {}
        try:
            color_fn: Callable[[str], str] = pastel_for_class
        except Exception:
            color_fn = lambda _cls: "#DDEBFF"
        self._aggr_color_fn = color_fn

        if prev_selection and prev_selection in values:
            self._aggr_selector.set(prev_selection)
            self._aggr_on_select()
        elif values:
            self._aggr_selector.current(0)
            self._aggr_on_select()
        else:
            self._aggr_redraw()
        self._aggr_refresh_side_lists()

    def _have_conf(self, name: str) -> bool:
        return (self._conf_dir_path / name).is_file()

    def _missing_conf_panel(self, parent: tk.Widget, name: str) -> None:
        frame = ttk.Frame(parent, padding=24)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            frame,
            text=f"⚠ '{name}' not found in '{self._conf_dir_path.name}'.",
            foreground="#b94a48",
            wraplength=360,
            justify=tk.CENTER,
        ).pack(pady=(0, 6))
        ttk.Label(
            frame,
            text="Create the file to enable this view.",
            wraplength=360,
            justify=tk.CENTER,
        ).pack()

    def _ui_globals(self) -> dict:
        data = try_load_globals()
        return data if isinstance(data, dict) else {}

    def _ui_pdvs(self) -> dict[str, dict]:
        data = try_load_pdvs()
        return data if isinstance(data, dict) else {}

    def _ui_classes(self) -> dict[str, dict]:
        data = try_load_classes()
        return data if isinstance(data, dict) else {}

    def _ui_agents(self) -> list[dict]:
        data = try_load_agents()
        return data if isinstance(data, list) else []

    def _restore_class_tab_selection(self) -> None:
        lst = getattr(self, "cls_list", None)
        if lst is None:
            return
        last = self._load_ui_state().get("last_class")
        if last and last in self._classes_map:
            items = list(lst.get(0, tk.END))
            if last in items:
                idx = items.index(last)
                lst.selection_clear(0, tk.END)
                lst.selection_set(idx)
        elif lst.size() > 0:
            lst.selection_clear(0, tk.END)
            lst.selection_set(0)
        if lst.size() > 0:
            self._on_class_select()

    def _handle_conf_presence_changes(self, changes: list[tuple[str, bool]]) -> None:
        for name, present in changes:
            if name == "globals.json":
                # Rebuild the Globals tab when the file is created/updated,
                # but never auto-open a blocking dialog.
                self._build_globals_tab()
            elif name == "pdvs.json":
                self._build_pdvs_tab()
            elif name == "classes.json":
                self._build_classes_tab()
                if present:
                    self._restore_class_tab_selection()
                self._refresh_agent_class_choices()
            elif name == "agents.json":
                if present:
                    self._ensure_agent_group_membership()
                self._build_agents_editor_tab()
                self._build_simple_groups_tab()

    def _aggr_on_select(self, _evt=None) -> None:
        sel = self._aggr_selector.get() if hasattr(self, "_aggr_selector") else ""
        self._aggr_current_value = sel
        if sel.startswith("Agent: "):
            name = sel[7:]
            self._aggr_mode = "agent"
            self._aggr_active_agent = self._aggr_agents_by_name.get(name)
            self._aggr_active_group = None
        elif sel.startswith("Group: "):
            grp = sel[7:]
            self._aggr_mode = "group"
            self._aggr_active_group = grp
            self._aggr_active_agent = None
        else:
            self._aggr_mode = None
            self._aggr_active_agent = None
            self._aggr_active_group = None
        self._aggr_refresh_side_lists()
        self._aggr_redraw()

    def _aggr_refresh_side_lists(self) -> None:
        mode = getattr(self, "_aggr_mode", None)
        agents = getattr(self, "_aggr_agents_by_name", {})
        cache = getattr(self, "_agents_cache", [])
        all_groups = sorted(
            set(g for a in cache for g in (a.get("groups_in", []) or []))
            | set(g for a in cache for g in (a.get("groups_out", []) or []))
        )

        left_items: list[str] = []
        right_items: list[str] = []
        left_label = "Left"
        right_label = "Right"

        if mode == "agent" and self._aggr_active_agent:
            ag = self._aggr_active_agent
            gin = set(ag.get("groups_in", []) or [])
            gout = set(ag.get("groups_out", []) or [])
            # Prepend the "Create New Group" sentinel to both sides when in Agent mode
            left_items = [_NEW_GROUP_LABEL] + [g for g in all_groups if g not in gin]
            right_items = [_NEW_GROUP_LABEL] + [g for g in all_groups if g not in gout]
            left_label = "Add to groups_in"
            right_label = "Add to groups_out"
        elif mode == "group" and self._aggr_active_group is not None:
            grp = self._aggr_active_group
            names = sorted(agents.keys())
            left_items = [
                n
                for n in names
                if grp not in (agents[n].get("groups_out", []) or [])
            ]
            right_items = [
                n
                for n in names
                if grp not in (agents[n].get("groups_in", []) or [])
            ]
            left_label = f"Agents missing {grp} in groups_out"
            right_label = f"Agents missing {grp} in groups_in"
        else:
            left_items = []
            right_items = []

        for lb, items in (
            (getattr(self, "_aggr_left_list", None), left_items),
            (getattr(self, "_aggr_right_list", None), right_items),
        ):
            if lb is None:
                continue
            lb.delete(0, tk.END)
            for s in items:
                lb.insert(tk.END, s)

        if hasattr(self, "_aggr_left_label"):
            self._aggr_left_label.config(text=left_label)
        if hasattr(self, "_aggr_right_label"):
            self._aggr_right_label.config(text=right_label)

        if hasattr(self, "_aggr_left_add"):
            self._aggr_left_add.state(["disabled"])
        if hasattr(self, "_aggr_right_add"):
            self._aggr_right_add.state(["disabled"])

    def _aggr_update_add_state(self, side: str) -> None:
        if side == "left" and hasattr(self, "_aggr_left_add"):
            if self._aggr_left_list.curselection():
                self._aggr_left_add.state(["!disabled"])
            else:
                self._aggr_left_add.state(["disabled"])
        elif side == "right" and hasattr(self, "_aggr_right_add"):
            if self._aggr_right_list.curselection():
                self._aggr_right_add.state(["!disabled"])
            else:
                self._aggr_right_add.state(["disabled"])

    def _aggr_add_left(self) -> None:
        if getattr(self, "_aggr_mode", None) == "agent" and self._aggr_active_agent:
            idx = self._aggr_left_list.curselection()
            if not idx:
                return
            sel = self._aggr_left_list.get(idx[0])
            # If the sentinel is selected, prompt for a new group name
            if sel == _NEW_GROUP_LABEL:
                name = simpledialog.askstring(
                    "Create New Group", "Enter new group name:", parent=self.root
                )
                if not name:
                    return
                grp = name.strip()
                if not grp:
                    return
            else:
                grp = sel
            cache = getattr(self, "_agents_cache", [])
            target = next(
                (a for a in cache if a.get("name") == self._aggr_active_agent.get("name")),
                None,
            )
            if target is None:
                return
            target.setdefault("groups_in", [])
            if grp not in target["groups_in"]:
                target["groups_in"].append(grp)
                save_agents(cache)
                self._build_simple_groups_tab()
                if hasattr(self, "_aggr_selector"):
                    self._aggr_selector.set(self._aggr_current_value)
                    self._aggr_on_select()
        elif getattr(self, "_aggr_mode", None) == "group" and self._aggr_active_group is not None:
            idx = self._aggr_left_list.curselection()
            if not idx:
                return
            agent_name = self._aggr_left_list.get(idx[0])
            grp = self._aggr_active_group
            cache = getattr(self, "_agents_cache", [])
            target = next(
                (a for a in cache if a.get("name") == agent_name),
                None,
            )
            if target is None:
                return
            target.setdefault("groups_out", [])
            if grp not in target["groups_out"]:
                target["groups_out"].append(grp)
                save_agents(cache)
                self._build_simple_groups_tab()
                if hasattr(self, "_aggr_selector"):
                    self._aggr_selector.set(self._aggr_current_value)
                    self._aggr_on_select()

    def _aggr_add_right(self) -> None:
        if getattr(self, "_aggr_mode", None) == "agent" and self._aggr_active_agent:
            idx = self._aggr_right_list.curselection()
            if not idx:
                return
            sel = self._aggr_right_list.get(idx[0])
            # If the sentinel is selected, prompt for a new group name
            if sel == _NEW_GROUP_LABEL:
                name = simpledialog.askstring(
                    "Create New Group", "Enter new group name:", parent=self.root
                )
                if not name:
                    return
                grp = name.strip()
                if not grp:
                    return
            else:
                grp = sel
            cache = getattr(self, "_agents_cache", [])
            target = next(
                (a for a in cache if a.get("name") == self._aggr_active_agent.get("name")),
                None,
            )
            if target is None:
                return
            target.setdefault("groups_out", [])
            if grp not in target["groups_out"]:
                target["groups_out"].append(grp)
                save_agents(cache)
                self._build_simple_groups_tab()
                if hasattr(self, "_aggr_selector"):
                    self._aggr_selector.set(self._aggr_current_value)
                    self._aggr_on_select()
        elif getattr(self, "_aggr_mode", None) == "group" and self._aggr_active_group is not None:
            idx = self._aggr_right_list.curselection()
            if not idx:
                return
            agent_name = self._aggr_right_list.get(idx[0])
            grp = self._aggr_active_group
            cache = getattr(self, "_agents_cache", [])
            target = next(
                (a for a in cache if a.get("name") == agent_name),
                None,
            )
            if target is None:
                return
            target.setdefault("groups_in", [])
            if grp not in target["groups_in"]:
                target["groups_in"].append(grp)
                save_agents(cache)
                self._build_simple_groups_tab()
                if hasattr(self, "_aggr_selector"):
                    self._aggr_selector.set(self._aggr_current_value)
                    self._aggr_on_select()

    def _aggr_remove_click(self, kind: str, name: str, side: str) -> None:
        cache = self._agents_cache
        mode = self._aggr_mode
        warn = lambda txt: messagebox.showwarning("Remove", txt)

        def _safe_pop(lst: list[str], item: str, label: str) -> bool:
            if item not in lst:
                return False
            # Allow empty lists; removing the last item is OK.
            lst.remove(item)
            return True

        changed = False

        if mode == "agent" and self._aggr_active_agent and kind == "group":
            target = next(
                (a for a in cache if a.get("name") == self._aggr_active_agent.get("name")),
                None,
            )
            if not target:
                return
            # left = incoming, right = outgoing (for an AGENT)
            lst = (
                target.setdefault("groups_in", [])
                if side == "left"
                else target.setdefault("groups_out", [])
            )
            if _safe_pop(lst, name, f"{target['name']} {side} list"):
                changed = True

        elif mode == "group" and self._aggr_active_group and kind == "agent":
            target = next((a for a in cache if a.get("name") == name), None)
            if not target:
                return
            grp = self._aggr_active_group
            # left = senders/outbound, right = listeners/inbound (for a GROUP)
            lst = (
                target.setdefault("groups_out", [])
                if side == "left"
                else target.setdefault("groups_in", [])
            )
            if _safe_pop(lst, grp, f"{target['name']} {side} list"):
                changed = True
        else:
            return

        if not changed:
            return

        try:
            save_agents(cache)
        except ValueError as exc:
            warn(str(exc))
            return

        self._build_simple_groups_tab()
        if hasattr(self, "_aggr_selector"):
            self._aggr_selector.set(self._aggr_current_value)
            self._aggr_on_select()

    def _ensure_agent_group_membership(self) -> None:
        if not self._have_conf("agents.json"):
            return
        agents = self._ui_agents()
        if not agents:
            return

        all_groups = sorted(
            {
                g
                for agent in agents
                for g in (agent.get("groups_in", []) or []) + (agent.get("groups_out", []) or [])
                if g
            }
        )
        changed = False

        for agent in agents:
            name = agent.get("name") or "Unnamed agent"
            missing_sides: list[tuple[str, str]] = []
            if not (agent.get("groups_in") or []):
                missing_sides.append(("groups_in", "incoming"))
            if not (agent.get("groups_out") or []):
                missing_sides.append(("groups_out", "outgoing"))

            for side_key, side_label in missing_sides:
                selections = self._prompt_agent_group_selection(name, side_label, all_groups)
                if not selections:
                    continue
                agent[side_key] = selections
                all_groups = sorted({*all_groups, *selections})
                changed = True

        if not changed:
            return

        try:
            save_agents(agents)
        except ValueError as exc:
            messagebox.showwarning("Groups", str(exc))
            return

    def _prompt_agent_group_selection(
        self, agent_name: str, side_label: str, choices: list[str]
    ) -> list[str]:
        title = "Assign Groups"
        prompt = (
            f"Agent '{agent_name}' is missing {side_label} groups.\n"
            "Select one or more existing groups or enter new group names (comma separated)."
        )
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(dialog, text=prompt, wraplength=360, justify=tk.LEFT).pack(
            anchor="w", padx=12, pady=(12, 8)
        )

        listbox = tk.Listbox(dialog, selectmode=tk.MULTIPLE, height=8, exportselection=False)
        for choice in choices:
            listbox.insert(tk.END, choice)
        listbox.pack(fill=tk.BOTH, expand=True, padx=12)

        entry_frame = ttk.Frame(dialog)
        entry_frame.pack(fill=tk.X, padx=12, pady=(8, 4))
        ttk.Label(entry_frame, text="New groups:").pack(side=tk.LEFT)
        entry_var = tk.StringVar()
        entry = ttk.Entry(entry_frame, textvariable=entry_var, width=35)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        result: list[str] = []

        def _apply() -> None:
            selected_indices = listbox.curselection()
            selected = [listbox.get(i) for i in selected_indices]
            new_raw = [part.strip() for part in entry_var.get().split(",") if part.strip()]
            combined: list[str] = []
            seen = set()
            for name in selected + new_raw:
                if name and name not in seen:
                    combined.append(name)
                    seen.add(name)
            if not combined:
                messagebox.showwarning(title, "Select or enter at least one group before continuing.")
                return
            result.extend(combined)
            dialog.destroy()

        def _block_close() -> None:
            messagebox.showwarning(title, "Please assign at least one group to continue.")

        dialog.protocol("WM_DELETE_WINDOW", _block_close)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=12, pady=(4, 12))
        ttk.Button(btn_frame, text="Assign", command=_apply).pack(side=tk.RIGHT)

        listbox.focus_set()
        self.root.wait_window(dialog)
        return result

    def _aggr_spread_y(self, n: int, height: int, margin: int = 36) -> list[int]:
        if n <= 0:
            return []
        usable = max(1, height - 2 * margin)
        if n == 1:
            return [margin + usable // 2]
        step = usable / (n - 1)
        return [int(margin + i * step) for i in range(n)]

    def _aggr_redraw(self) -> None:
        c = getattr(self, "_aggr_canvas", None)
        if not c:
            return
        c.delete("all")
        if hasattr(self, "_aggr_node_items"):
            self._aggr_node_items.clear()
        w = max(1, int(c.winfo_width()))
        h = max(1, int(c.winfo_height()))
        cx = w // 2
        left_x, right_x = int(w * 0.2), int(w * 0.8)

        if self._aggr_mode == "agent" and self._aggr_active_agent:
            ag = self._aggr_active_agent
            gin = sorted(set(ag.get("groups_in", []) or []))
            gout = sorted(set(ag.get("groups_out", []) or []))
            ys_l = self._aggr_spread_y(len(gin), h)
            ys_r = self._aggr_spread_y(len(gout), h)
            color_fn = getattr(self, "_aggr_color_fn", lambda _cls: "#DDEBFF")
            for y, g in zip(ys_l, gin):
                self._aggr_draw_group_node(left_x, y, g, side="left")
                self._aggr_arrow(left_x + 40, y, cx - 24, h // 2)
            for y, g in zip(ys_r, gout):
                self._aggr_draw_group_node(right_x, y, g, side="right")
                self._aggr_arrow(cx + 24, h // 2, right_x - 40, y)
            self._aggr_draw_agent_node(cx, h // 2, ag, color_fn)
        elif self._aggr_mode == "group" and self._aggr_active_group is not None:
            grp = self._aggr_active_group
            senders = getattr(self, "_aggr_groups_senders", {}).get(grp, [])
            listeners = getattr(self, "_aggr_groups_listeners", {}).get(grp, [])
            ys_l = self._aggr_spread_y(len(senders), h)
            ys_r = self._aggr_spread_y(len(listeners), h)
            color_fn = getattr(self, "_aggr_color_fn", lambda _cls: "#DDEBFF")
            for y, name in zip(ys_l, senders):
                agent = self._aggr_agents_by_name.get(name)
                if agent:
                    self._aggr_draw_agent_node(
                        left_x, y, agent, color_fn, r=18, side="left"
                    )
                    self._aggr_arrow(left_x + 20, y, cx - 48, h // 2)
            for y, name in zip(ys_r, listeners):
                agent = self._aggr_agents_by_name.get(name)
                if agent:
                    self._aggr_draw_agent_node(
                        right_x, y, agent, color_fn, r=18, side="right"
                    )
                    self._aggr_arrow(cx + 48, h // 2, right_x - 20, y)
            self._aggr_draw_group_node(
                cx,
                h // 2,
                grp,
                center=True,
                counts=(len(senders), len(listeners)),
            )
        else:
            c.create_text(cx, h // 2, text="Select an Agent or Group")

    def _aggr_draw_agent_node(
        self,
        x: int,
        y: int,
        agent: dict,
        color_fn: Callable[[str], str],
        r: int = 24,
        side: Optional[str] = None,
    ) -> None:
        cls_name = agent.get("agent_class") or agent.get("role", "")
        color = color_fn(cls_name)
        circle = self._aggr_canvas.create_oval(
            x - r,
            y - r,
            x + r,
            y + r,
            fill=color,
            outline="black",
        )
        name = agent.get("name", "")
        display = name if len(name) <= 18 else name[:17] + "…"
        text = self._aggr_canvas.create_text(x, y + r + 12, text=display)
        for item in (circle, text):
            self._aggr_canvas.tag_bind(
                item,
                "<Enter>",
                lambda e, a=agent: self._show_tooltip(e.x_root, e.y_root, a),
            )
            self._aggr_canvas.tag_bind(item, "<Leave>", lambda _e: self._hide_tooltip())
            # Single-click recenters on this agent. Keep double-click for parity.
            self._aggr_canvas.tag_bind(item, "<Button-1>", lambda _e, n=name: self._aggr_select(f"Agent: {n}"))
            self._aggr_canvas.tag_bind(item, "<Double-1>", lambda _e, n=name: self._aggr_select(f"Agent: {n}"))
            self._aggr_canvas.tag_bind(
                item,
                "<Button-3>",
                lambda _e, n=name, s=side: self._aggr_remove_click("agent", n, s)
                if s in ("left", "right")
                else None,
            )
            self._aggr_node_items[item] = f"Agent: {name}"

    def _aggr_draw_group_node(
        self,
        x: int,
        y: int,
        group: str,
        center: bool = False,
        counts: Optional[tuple[int, int]] = None,
        side: Optional[str] = None,
    ) -> None:
        senders = list(self._aggr_groups_senders.get(group, []))
        listeners = list(self._aggr_groups_listeners.get(group, []))
        w, h = (90, 26) if not center else (120, 34)
        rect = self._aggr_canvas.create_rectangle(
            x - w // 2,
            y - h // 2,
            x + w // 2,
            y + h // 2,
            fill="#FFF8D6",
            outline="black",
        )
        label = group if len(group) <= 22 else group[:21] + "…"
        if center:
            left = counts[0] if counts else len(senders)
            right = counts[1] if counts else len(listeners)
            label = f"{group}  [{left}→  ←{right}]"
            if len(label) > 30:
                label = label[:29] + "…"
        text = self._aggr_canvas.create_text(x, y, text=label)
        info = {
            "name": group,
            "role": "Group",
            "groups_out": senders,
            "groups_in": listeners,
        }
        for item in (rect, text):
            self._aggr_canvas.tag_bind(
                item,
                "<Enter>",
                lambda e, a=info: self._show_tooltip(e.x_root, e.y_root, a),
            )
            self._aggr_canvas.tag_bind(item, "<Leave>", lambda _e: self._hide_tooltip())
            # Single-click recenters on this group. Keep double-click for parity.
            self._aggr_canvas.tag_bind(item, "<Button-1>", lambda _e, g=group: self._aggr_select(f"Group: {g}"))
            self._aggr_canvas.tag_bind(item, "<Double-1>", lambda _e, g=group: self._aggr_select(f"Group: {g}"))
            self._aggr_canvas.tag_bind(
                item,
                "<Button-3>",
                lambda _e, g=group, s=side: self._aggr_remove_click("group", g, s)
                if s in ("left", "right")
                else None,
            )
            self._aggr_node_items[item] = f"Group: {group}"

    def _aggr_arrow(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self._aggr_canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST)

    def _aggr_select(self, value: str) -> None:
        if not hasattr(self, "_aggr_selector"):
            return
        vals = list(self._aggr_selector.cget("values"))
        if value in vals:
            self._aggr_selector.set(value)
            self._aggr_on_select()

    def _on_flag_select(self, _event=None) -> None:
        sel = getattr(self, "_flag_list", None)
        if not sel:
            return
        idx = sel.curselection()
        if not idx:
            return
        name = sel.get(idx[0])
        agent = next((a for a in self._agents_cache if a["name"] == name), None)
        if not agent:
            return
        lines = []
        for g in agent.get("groups_out", []):
            consumers = [b["name"] for b in self._agents_cache if b["name"] != name and g in b.get("groups_in", [])]
            if consumers:
                lines.append(f"{g} -> {', '.join(consumers)}")
            else:
                lines.append(f"{g} -> (no consumers)")
        self._flag_info.config(text="\n".join(lines))

    def _batch_wiring(self) -> None:
        if not hasattr(self, "_flag_list"):
            return
        idx = self._flag_list.curselection()
        if not idx:
            messagebox.showinfo("Batch Wiring", "Select a flagged agent first")
            return
        name = self._flag_list.get(idx[0])
        agent = next(a for a in self._agents_cache if a["name"] == name)
        missing = agent.get("missing_out_groups") or [
            g for g in agent.get("groups_out", [])
            if not any(g in b.get("groups_in", []) for b in self._agents_cache if b["name"] != name)
        ]
        if not missing:
            messagebox.showinfo("Batch Wiring", "No missing groups")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Batch Wiring")
        selections = []
        for g in missing:
            ttk.Label(dialog, text=f"{g}:").pack(anchor="w")
            cands = [b for b in self._agents_cache if b["name"] != name and g not in b.get("groups_in", [])]
            for cand in cands:
                var = tk.BooleanVar()
                chk = tk.Checkbutton(dialog, text=cand["name"], variable=var)
                chk.pack(anchor="w", padx=20)
                selections.append((var, cand["name"], g))
        def _apply() -> None:
            import copy, json
            new_agents = copy.deepcopy(self._agents_cache)
            affected = set()
            for var, cand_name, grp in selections:
                if var.get():
                    targ = next(a for a in new_agents if a["name"] == cand_name)
                    targ.setdefault("groups_in", [])
                    if grp not in targ["groups_in"]:
                        targ["groups_in"].append(grp)
                        affected.add(cand_name)
            if not affected:
                dialog.destroy()
                return
            targ_agent = next(a for a in new_agents if a["name"] == name)
            targ_agent["flag_no_downstream"] = False
            targ_agent["missing_out_groups"] = []
            affected.add(name)
            preview = {a["name"]: a for a in new_agents if a["name"] in affected}
            prev_win = tk.Toplevel(dialog)
            prev_win.title("Preview Changes")
            txt = scrolledtext.ScrolledText(prev_win)
            txt.pack(fill=tk.BOTH, expand=True)
            txt.insert("1.0", json.dumps(preview, indent=2))
            def _confirm() -> None:
                save_agents(new_agents)
                prev_win.destroy()
                dialog.destroy()
                self._build_simple_groups_tab()
            ttk.Button(prev_win, text="Apply", command=_confirm).pack()
        ttk.Button(dialog, text="Apply", command=_apply).pack(pady=4)

    def _repair_dead_ends(self) -> None:
        import copy, json
        new_agents = copy.deepcopy(self._agents_cache)
        affected = set()
        for agent in new_agents:
            if agent.get("flag_no_downstream"):
                missing = agent.get("missing_out_groups") or [
                    g for g in agent.get("groups_out", [])
                    if not any(g in b.get("groups_in", []) for b in new_agents if b["name"] != agent["name"])
                ]
                for g in missing:
                    cands = [b for b in new_agents if b["name"] != agent["name"] and g not in b.get("groups_in", [])]
                    if not cands:
                        continue
                    targ = sorted(cands, key=lambda x: x["name"])[0]
                    targ.setdefault("groups_in", [])
                    if g not in targ["groups_in"]:
                        targ["groups_in"].append(g)
                        affected.add(targ["name"])
                agent["flag_no_downstream"] = False
                agent["missing_out_groups"] = []
                affected.add(agent["name"])
        if not affected:
            messagebox.showinfo("Repair Dead-Ends", "No changes proposed")
            return
        preview = {a["name"]: a for a in new_agents if a["name"] in affected}
        win = tk.Toplevel(self.root)
        win.title("Preview Repair")
        txt = scrolledtext.ScrolledText(win)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", json.dumps(preview, indent=2))
        def _apply() -> None:
            save_agents(new_agents)
            win.destroy()
            self._build_simple_groups_tab()
        ttk.Button(win, text="Apply", command=_apply).pack()

    def _fetch_models(self) -> list[str]:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m.get("name") for m in resp.json().get("models", [])]
            self._model_cache = sorted([m for m in models if m])
        except Exception as exc:
            logger.exception("Model list fetch failed")
            messagebox.showwarning("Models", f"Failed to fetch models: {exc}")
        return list(self._model_cache)

    def _ensure_globals_set(self) -> None:
        """
        Only prompt for Globals if globals.json exists and is non-empty,
        but missing required keys. An absent or empty file should NOT
        open the dialog; tabs stay accessible/blank.
        """
        p = self._conf_dir_path / "globals.json"
        if not p.exists():
            return
        try:
            # Treat 0–2 byte files (empty or "{}") as blank configs.
            if p.stat().st_size <= 2:
                return
        except Exception:
            return  # Be conservative; never block on errors.

        data = try_load_globals()
        if not isinstance(data, dict):
            return

        # Update in-memory config so the dialog shows current values if needed.
        self.global_config = data

        need = (not data.get("model")) or (data.get("temperature") is None)
        if need:
            self._open_globals_dialog_blocking()

    def _open_globals_dialog_blocking(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Set Globals")
        dlg.transient(self.root)
        vars = {
            "debug_level": tk.StringVar(value=self.global_config.get("debug_level", "INFO")),
            "model": tk.StringVar(value=self.global_config.get("model", "")),
            "temperature": tk.StringVar(value=str(self.global_config.get("temperature", ""))),
            "system_prompt": tk.StringVar(value=self.global_config.get("system_prompt", "")),
            "pre_context_message": tk.StringVar(value=self.global_config.get("pre_context_message", "")),
            "post_context_message": tk.StringVar(value=self.global_config.get("post_context_message", "")),
            "max_context_tokens": tk.StringVar(value=str(self.global_config.get("max_context_tokens", 8192))),
            "pdv_gamma": tk.StringVar(value=str(self.global_config.get("pdv_gamma", 2.0))),
            "watchdog_timeout": tk.StringVar(value=str(self.global_config.get("watchdog_timeout", 900))),
        }
        models = self._fetch_models()
        row = 0
        for label, key in [
            ("Debug Level", "debug_level"),
            ("Model", "model"),
            ("Temperature", "temperature"),
            ("System Prompt", "system_prompt"),
            ("Pre Context", "pre_context_message"),
            ("Post Context", "post_context_message"),
            ("Max Tokens", "max_context_tokens"),
            ("PDV Gamma", "pdv_gamma"),
            ("Watchdog Timeout (s)  (0=disabled)", "watchdog_timeout"),
        ]:
            tk.Label(dlg, text=label).grid(row=row, column=0, sticky="w")
            if key == "model":
                box = ttk.Combobox(dlg, textvariable=vars[key], values=models, state="readonly")
                box.grid(row=row, column=1, sticky="ew")
            else:
                entry = ttk.Entry(dlg, textvariable=vars[key])
                entry.grid(row=row, column=1, sticky="ew")
            row += 1
        dlg.columnconfigure(1, weight=1)
        btn = ttk.Frame(dlg)
        btn.grid(row=row, column=0, columnspan=2, pady=4)

        def _refresh() -> None:
            box.configure(values=self._fetch_models())

        def _save() -> None:
            temp_cfg = {}
            for k, var in vars.items():
                val = var.get()
                if k in {"temperature", "max_context_tokens", "pdv_gamma", "watchdog_timeout"}:
                    try:
                        if k == "max_context_tokens":
                            temp_cfg[k] = int(val)
                        elif k == "watchdog_timeout":
                            temp_cfg[k] = int(float(val))
                        else:
                            temp_cfg[k] = float(val)
                    except ValueError:
                        messagebox.showerror("Globals", f"Invalid value for {k}")
                        return
                else:
                    temp_cfg[k] = val
            if not temp_cfg.get("model") or temp_cfg.get("temperature") is None:
                messagebox.showerror(
                    "Globals", "Model and temperature are required and must be valid"
                )
                return
            self.global_config.update(temp_cfg)
            save_globals(self.global_config)
            self.global_config = self._ui_globals()
            self._build_globals_tab()
            self.base_timeout = self.global_config.get("watchdog_timeout", 900)
            txt = (
                "Base Timeout: disabled"
                if (self.base_timeout is None or float(self.base_timeout) <= 0)
                else f"Base Timeout: {int(self.base_timeout)}s"
            )
            self.timeout_label.config(text=txt)
            # Live-apply here as well
            try:
                if self.on_apply_globals:
                    self.on_apply_globals(dict(self.global_config))
            except Exception:
                logger.exception("Failed to apply globals update callback (dialog)")
            dlg.grab_release()
            dlg.destroy()

        ttk.Button(btn, text="Refresh Models", command=_refresh).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn, text="Save", command=_save).pack(side=tk.LEFT, padx=2)

        dlg.grab_set()
        self.root.wait_window(dlg)

    def _start_metrics_poll(self):
        def _tick():
            try:
                with open(
                    os.path.join("chatlogs", "pdvs_live.json"), "r", encoding="utf-8"
                ) as f:
                    pdv_vals = json.load(f)
                if isinstance(pdv_vals, dict):
                    # If new PDVs appear, rebuild rows first
                    if set(pdv_vals.keys()) != set(self.pdv_values.keys()):
                        self.pdv_values = dict(pdv_vals)
                        self._rebuild_live_metrics_rows()
                    self.update_pdvs(pdv_vals)
            except Exception:
                pass
            try:
                with open(
                    os.path.join("chatlogs", "token_usage.json"), "r", encoding="utf-8"
                ) as f:
                    info = json.load(f)
                self.set_token_usage(int(info.get("used", 0)), int(info.get("limit", 0)))
            except Exception:
                pass
            self.root.after(2000, _tick)

        self.root.after(2000, _tick)


    def set_active_agent(self, name: str) -> None:
        logger.debug("Entering set_active_agent name=%s", name)

        def _update():
            self._active_agent = name or None
            display = name or "None"
            self.agent_header.config(text=f"Current Agent: {display}")
            payload = self._agent_payloads.get(name) if name else ""
            self._render_agent_payload(payload)

        self._threadsafe(_update)
        logger.debug("Exiting set_active_agent")

    def update_agent_payload(self, agent: str, payload: dict) -> None:
        logger.debug(
            "Entering update_agent_payload agent=%s keys=%s",
            agent,
            list(payload.keys()),
        )
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        self._agent_payloads[agent] = text

        def _update():
            if self._active_agent == agent:
                self._render_agent_payload(text)

        self._threadsafe(_update)
        logger.debug("Exiting update_agent_payload")

    def set_group_contexts(self, context_by_group: dict[str, str]) -> None:
        logger.debug(
            "Entering set_group_contexts context_by_group_keys=%s",
            list(context_by_group.keys()),
        )

        def _update():
            current = None
            sel = self.group_list.curselection()
            if sel:
                current = self.group_list.get(sel[0])
            self._group_contexts = dict(context_by_group)
            self.group_list.delete(0, tk.END)
            for grp in self._group_contexts:
                self.group_list.insert(tk.END, grp)
            target = current if current in self._group_contexts else None
            if not target and self._group_contexts:
                target = next(iter(self._group_contexts))
                idx = list(self._group_contexts).index(target)
                self.group_list.selection_set(idx)
            if target:
                self._render_group_text(self._group_contexts.get(target, ""))
            else:
                self._render_group_text("")

        self._threadsafe(_update)
        logger.debug("Exiting set_group_contexts")

    def update_group_context(self, group: str, text: str) -> None:
        logger.debug("Entering update_group_context group=%s", group)

        def _update():
            self._group_contexts[group] = text
            names = list(self.group_list.get(0, tk.END))
            if group not in names:
                self.group_list.insert(tk.END, group)
            sel = self.group_list.curselection()
            if sel and self.group_list.get(sel[0]) == group:
                self._render_group_text(text)

        self._threadsafe(_update)
        logger.debug("Exiting update_group_context")

    def _on_group_select(self, _event=None):
        sel = self.group_list.curselection()
        if not sel:
            return
        group = self.group_list.get(sel[0])
        text = self._group_contexts.get(group, "")
        self._render_group_text(text)

    def _render_agent_payload(self, text: str | None) -> None:
        self.agent_payload_view.configure(state="normal")
        self.agent_payload_view.delete("1.0", tk.END)
        if text:
            self.agent_payload_view.insert(tk.END, text)
            lines = self.agent_payload_view.get("1.0", tk.END).splitlines()
            if len(lines) > 2000:
                trimmed = "\n".join(lines[-2000:])
                self.agent_payload_view.delete("1.0", tk.END)
                self.agent_payload_view.insert(tk.END, trimmed)
        self.agent_payload_view.configure(state="disabled")
        self.agent_payload_view.see(tk.END)

    def _render_group_text(self, text: str) -> None:
        self.group_text.configure(state="normal")
        self.group_text.delete("1.0", tk.END)
        if text:
            self.group_text.insert(tk.END, text)
        self.group_text.configure(state="disabled")
        self.group_text.see(tk.END)

    def _copy_agent_payload(self) -> None:
        logger.debug("Entering _copy_agent_payload")
        text = self.agent_payload_view.get("1.0", tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        logger.debug("Exiting _copy_agent_payload")

    def _save_agent_payload(self) -> None:
        logger.debug("Entering _save_agent_payload")
        text = self.agent_payload_view.get("1.0", tk.END)
        if not text.strip():
            logger.debug("_save_agent_payload called with empty text")
            return
        fname = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if fname:
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(text)
            except OSError as exc:  # noqa: BLE001
                logger.error("Failed to save JSON payload: %s", exc)
        logger.debug("Exiting _save_agent_payload")

    def _clear_agent_payload(self) -> None:
        logger.debug("Entering _clear_agent_payload")
        self._render_agent_payload("")
        logger.debug("Exiting _clear_agent_payload")

    def update_topology(self, active_agent: dict, agents: list[dict]) -> None:
        logger.debug("Entering update_topology active_agent=%s agents=%s", active_agent, agents)

        def _update():
            self._topology_active = active_agent
            self._topology_agents = agents
            self._redraw_topology()

        self._threadsafe(_update)
        logger.debug("Exiting update_topology")

    def _compute_neighbors(self, active: dict, agents: list[dict]) -> tuple[list[dict], list[dict], int, int]:
        upstream: list[dict] = []
        downstream: list[dict] = []
        active_in = set(active.get("groups_in", []) or [])
        active_out = set(active.get("groups_out", []) or [])
        for ag in agents:
            if ag.get("name") == active.get("name"):
                continue
            ag_in = set(ag.get("groups_in", []) or [])
            ag_out = set(ag.get("groups_out", []) or [])
            if ag_out & active_in:
                upstream.append(ag)
            if active_out & ag_in:
                downstream.append(ag)
        extra_up = max(0, len(upstream) - 25)
        extra_down = max(0, len(downstream) - 25)
        return upstream[:25], downstream[:25], extra_up, extra_down

    def _redraw_topology(self) -> None:
        active = self._topology_active
        self._hide_tooltip()
        self.topology_canvas.delete("all")
        if not active:
            self.topology_header.config(text="Active Agent: None")
            return
        cls = active.get('agent_class') or active.get('role', '')
        self.topology_header.config(
            text=f"Active Agent: {active.get('name')} ({cls.title()})"
        )
        agents = self._topology_agents
        upstream, downstream, extra_up, extra_down = self._compute_neighbors(active, agents)
        width = self.topology_canvas.winfo_width() or 1
        height = self.topology_canvas.winfo_height() or 1
        cx_up, cx_act, cx_down = width * 0.2, width * 0.5, width * 0.8
        active_y = height / 2

        def positions(n: int) -> list[float]:
            pad = 40
            if n <= 0:
                return []
            step = (height - pad * 2) / n
            return [pad + step / 2 + i * step for i in range(n)]

        up_pos = positions(len(upstream))
        down_pos = positions(len(downstream))
        self._topology_node_items.clear()

        for ag, y in zip(upstream, up_pos):
            self._draw_node(cx_up, y, ag, 16)
            self._draw_arrow(cx_up, y, cx_act, active_y)
        if extra_up:
            self.topology_canvas.create_text(cx_up, height - 20, text=f"+{extra_up} more…")
        elif not upstream:
            self.topology_canvas.create_text(
                cx_up, active_y, text="No likely sources", fill="#888888"
            )

        for ag, y in zip(downstream, down_pos):
            self._draw_node(cx_down, y, ag, 16)
            self._draw_arrow(cx_act, active_y, cx_down, y)
        if extra_down:
            self.topology_canvas.create_text(cx_down, height - 20, text=f"+{extra_down} more…")
        elif not downstream:
            self.topology_canvas.create_text(
                cx_down, active_y, text="No likely targets", fill="#888888"
            )

        self._draw_node(cx_act, active_y, active, 24)

    def _draw_node(self, x: float, y: float, agent: dict, radius: int) -> None:
        cls_name = agent.get("agent_class") or agent.get("role", "")
        color = pastel_for_class(cls_name)
        circle = self.topology_canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=color,
            outline="black",
        )
        name = agent.get("name", "")
        display = name if len(name) <= 18 else name[:17] + "…"
        text = self.topology_canvas.create_text(x, y + radius + 12, text=display)
        for item in (circle, text):
            self.topology_canvas.tag_bind(
                item,
                "<Enter>",
                lambda e, a=agent: self._show_tooltip(e.x_root, e.y_root, a),
            )
            self.topology_canvas.tag_bind(item, "<Leave>", lambda e: self._hide_tooltip())
            self.topology_canvas.tag_bind(
                item,
                "<Double-1>",
                lambda e, a=agent: self.update_topology(a, self._topology_agents),
            )
            self._topology_node_items[item] = agent

    def _draw_arrow(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.topology_canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST)

    def _show_tooltip(self, x: int, y: int, agent: dict) -> None:
        self._hide_tooltip()
        tip = tk.Toplevel(self.root)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x + 10}+{y + 10}")
        info = (
            f"{agent.get('name')}\nRole: {agent.get('role')}\n"
            f"in: {len(agent.get('groups_in', []))}  out: {len(agent.get('groups_out', []))}"
        )
        ttk.Label(tip, text=info, relief=tk.SOLID, borderwidth=1, padding=2).pack()
        self._topology_tooltip = tip

    def _hide_tooltip(self) -> None:
        if self._topology_tooltip:
            self._topology_tooltip.destroy()
            self._topology_tooltip = None

    def _append_text(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state="disabled")

    def _threadsafe(self, func, *args, **kwargs) -> None:
        if threading.current_thread() is threading.main_thread():
            func(*args, **kwargs)
        else:
            self.root.after(0, lambda: func(*args, **kwargs))

    def _expand_all(self):
        logger.debug("_expand_all called but tree view removed")

    def _collapse_all(self):
        logger.debug("_collapse_all called but tree view removed")

    def _send_from_box(self):
        logger.debug("_send_from_box called but message box removed")

    def _refresh_chat_display(self):
        logger.debug("_refresh_chat_display called but chat display removed")

    def _refresh_log_display(self):
        logger.debug("Entering _refresh_log_display")

        def _update():
            self.thought_stream.configure(state="normal")
            self.thought_stream.delete("1.0", tk.END)
            for m in self.log_messages:
                text = f"[{m['timestamp']}] {m['sender']}: {m['message']}\n{'-'*80}\n\n"
                self.thought_stream.insert(tk.END, text)
            self.thought_stream.configure(state="disabled")
            self.thought_stream.see(tk.END)

        self._threadsafe(_update)
        logger.debug("Exiting _refresh_log_display")

    def log(self, entry):
        logger.debug("Entering log entry=%s", entry)
        self.log_messages.append(entry)
        text = f"[{entry['timestamp']}] {entry['sender']}: {entry['message']}\n{'-'*80}\n\n"
        self._threadsafe(self._append_text, self.thought_stream, text)
        logger.debug("Exiting log")

    def start(self):
        logger.debug("Entering start")
        loop = asyncio.new_event_loop()

        def _run_loop():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_discord_in_ui())
            loop.run_forever()

        t = threading.Thread(target=_run_loop, daemon=True)
        t.start()
        try:
            self.root.mainloop()
        finally:
            self._stop_config_watcher()
            if loop.is_running():
                loop.call_soon_threadsafe(lambda: asyncio.create_task(stop_discord_in_ui()))
                loop.call_soon_threadsafe(loop.stop)
            t.join()
        logger.debug("Exiting start")
    def _rebuild_live_metrics_rows(self) -> None:
        """Rebuild the Live Metrics pie chart container."""

        def _do():
            if not getattr(self, "metrics_rows", None):
                return
            for child in self.metrics_rows.winfo_children():
                child.destroy()

            self.metric_bars.clear()
            self.metric_labels.clear()

            container = ttk.Frame(self.metrics_rows)
            container.pack(fill=tk.BOTH, expand=True)

            # Left: pie canvas
            self.metrics_canvas = tk.Canvas(container, background="white", highlightthickness=0)
            self.metrics_canvas.grid(row=0, column=0, sticky="nsew")

            # Right: legend
            self.metrics_legend = ttk.Frame(container)
            self.metrics_legend.grid(row=0, column=1, sticky="ns", padx=(8, 4))

            container.columnconfigure(0, weight=1)
            container.rowconfigure(0, weight=1)

            def _on_resize(_evt=None):
                self._draw_pdv_pie(self.pdv_values)

            self.metrics_canvas.bind("<Configure>", _on_resize)
            self._draw_pdv_pie(self.pdv_values)

        self._threadsafe(_do)

    def _color_for_pdv(self, name: str) -> str:
        # Reuse existing color helpers
        try:
            return pastel_for_class(name)
        except Exception:
            return "#cccccc"

    def _draw_pdv_pie(self, pdv_values: dict[str, float]) -> None:
        if not self.metrics_canvas or not self.metrics_canvas.winfo_exists():
            return

        c = self.metrics_canvas
        c.delete("all")

        # Guard: avoid negatives; treat missing/NaN as 0
        names = sorted(pdv_values.keys())
        vals = [max(0.0, float(pdv_values.get(n, 0.0) or 0.0)) for n in names]
        total = sum(vals)

        w = c.winfo_width() or 1
        h = c.winfo_height() or 1
        size = min(w, h) - 16
        size = max(size, 10)
        cx, cy = w // 2, h // 2
        r = size // 2
        bbox = (cx - r, cy - r, cx + r, cy + r)

        # Legend
        for child in (self.metrics_legend.winfo_children() if self.metrics_legend else []):
            child.destroy()

        if total <= 0.0:
            c.create_text(cx, cy, text="No PDV values", font=("TkDefaultFont", 10))
            return

        angle = 0.0
        for name, value in zip(names, vals):
            if value <= 0:
                continue
            frac = value / total
            extent = frac * 360.0
            color = self._color_for_pdv(name)

            # Slice
            c.create_arc(bbox, start=angle, extent=extent, fill=color, outline="")
            # Label position — center of the slice
            mid = math.radians(angle + extent / 2.0)
            rx = cx + int(r * 0.55 * math.cos(mid))
            ry = cy - int(r * 0.55 * math.sin(mid))  # y inverted in canvas
            c.create_text(rx, ry, text=f"{value:.2f}", font=("TkDefaultFont", 9, "bold"))

            # Legend row
            if self.metrics_legend:
                row = ttk.Frame(self.metrics_legend)
                swatch = tk.Canvas(row, width=14, height=14, highlightthickness=0, bg=color)
                swatch.pack(side=tk.LEFT, padx=(0, 6))
                ttk.Label(row, text=f"{name}").pack(side=tk.LEFT)
                ttk.Label(row, text=f"{value:.2f}", foreground="#555").pack(side=tk.LEFT, padx=(6, 0))
                row.pack(anchor="w", pady=2)

            angle += extent
