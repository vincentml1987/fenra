"""fenra.py - worlds-rebuild branch, genuinely blank rebuild.

Teddy's call (2026-09-08): back up from the fenra.py architecture on
fenras-aletheosis entirely and rebuild from a simpler foundation -
voices and groups as the only concepts, nothing else yet. Goal for
this first pass, his own words: "let the internal thoughts work" -
prove the core loop (a voice thinks, its thought reaches whoever
should see it) before anything gets layered on top. No functions, no
permissions, no Hearth, no Topology - those come later, deliberately
not here yet.

THE MODEL, exactly as specified:

- World (renamed from "session") - a fully separate container. Worlds
  share nothing with each other - no cross-world storage of any kind.
  Lives at worlds/<world>/.
- Voice - model, identity, context, currency. `behavior` existed in the
  first pass and is gone (2026-09-09) - it was the same boilerplate for
  every voice, and the HUD below (ending in identity) replaces what it
  was doing. `currency` (2026-09-09, default 10.0) is a real, if simple,
  balance any voice can move via `give_currency` - genuinely
  exploratory, no plan for it beyond seeing what they do with it once
  they can see it and move it. context is a single free-text field,
  fully editable by Teddy at any time ("even context," his words) - not
  a fixed-size window, not a separate history file. It grows by plain
  string append: every time a voice thinks, and every time a fellow
  group member's thought lands, one line gets appended: "[timestamp]
  name: text" - the SAME format whether it's the voice's own thought or
  an incoming one, no special case for self vs. other.
- Group - exactly two fields: name, members (a list of voice names).
  No owner, no join_policy, no visibility, no direction - manually
  managed only, entirely from the GUI. No voice-driven join/invite/kick
  since there are no functions yet for a voice to call at all.

THE LOOP: one voice per tick, simple round-robin across the world's
voice list. Build the prompt as context + HUD (`build_hud()`), call
Ollama, get the raw response, run it through `run_function_calls()`
(2026-09-09 - functions are back, no permission layer this round, every
voice can call everything). That returns two versions of the response
(2026-09-10): the real one, with any `⟦function_name(args)⟧` calls
resolved into `⟦RESULT: ...⟧` text folded in - that's what the speaker's
own context gets - and a masked one, where every call becomes a plain
"(*speaker called name*)" notice with no arguments and no result at all
- that's what every OTHER member of every group the speaker belongs to
gets instead (deduped across overlapping groups). A voice can always see
what it did; bystanders only see that something happened, not what.

THE HUD: the last thing in every prompt, computed fresh every tick and
never persisted to context (Teddy's call, 2026-09-09 - it reflects live
world state and shouldn't compound the same context-bloat problem a
silently-timing-out voice can already produce). Tells a voice its own
name/model, its own groups, every group that exists in the world, who
it can currently see (shares a group with), who exists but isn't
visible to it, *everyone's* currency balance - not just its own
(2026-09-10, Teddy's call: full transparency, deliberately with no goal
attached, after watching give_currency turn into rote/formulaic use -
see whether visibility on its own changes anything) - and how to call/
discover functions (hard-coded, same reasoning as the old branch's
bootstrap notice - the calling convention is mechanics, not content, so
it isn't optional) - ending with its own identity line as the literal
last line of the entire prompt.

FUNCTIONS: reintroduced 2026-09-09, using the old branch's exact
`⟦function_name(args)⟧` call syntax (U+27E6/U+27E7 - essentially never
appears by accident) and `FUNCTION_REGISTRY` shape, but rebuilt lean -
no permission layer (every voice can call everything), no
`functions.jsonl` logging, no fabrication-detection. `send_message`
delivers straight into the target's real `context` via the existing
`append_to_context`, wrapped in an explicit flag so it reads as a
message rather than ordinary group chatter - not a separate, transitory
mechanism. `give_currency` moves real balance between two voices'
`currency` fields. `functions()` lists what's callable.
"""

import json
import os
import re
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, scrolledtext, simpledialog, ttk

import requests

FENRA_VERSION = "0.1.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORLDS_DIR = os.path.join(BASE_DIR, "worlds")

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3"
DEFAULT_INTERVAL_SEC = 3
DEFAULT_WORLD_NAME = "default"
DEFAULT_VOICE_NAME = "voice1"
CURRENCY_REFRESH_MS = 10000

WORLD_STATE_FILENAME = "world.json"
VOICE_STATE_FILENAME = "state.json"
START_SIGNAL_FILENAME = "start_signal.txt"
STOP_SIGNAL_FILENAME = "stop_signal.txt"

_GROUP_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


# --------------------------------------------------------------- storage --

def sanitize_name(name):
    return (name or "").strip().lower().replace(" ", "_").replace("'", "")


def world_dir(world_name):
    return os.path.join(WORLDS_DIR, world_name)


def ensure_world_dir(world_name):
    path = world_dir(world_name)
    os.makedirs(path, exist_ok=True)
    return path


def list_worlds():
    if not os.path.isdir(WORLDS_DIR):
        return []
    names = [d for d in os.listdir(WORLDS_DIR) if os.path.isdir(os.path.join(WORLDS_DIR, d))]

    def sort_key(name):
        path = os.path.join(world_dir(name), WORLD_STATE_FILENAME)
        return os.path.getmtime(path) if os.path.exists(path) else 0

    return sorted(names, key=sort_key, reverse=True)


def default_world_state():
    return {
        "host": DEFAULT_HOST,
        "interval": DEFAULT_INTERVAL_SEC,
        "model_default": DEFAULT_MODEL,
        "voices": [],
        "voice_rotation_index": 0,
    }


def world_state_path(world_name):
    return os.path.join(world_dir(world_name), WORLD_STATE_FILENAME)


def load_world_state(world_name):
    state = default_world_state()
    path = world_state_path(world_name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                state.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return state


def save_world_state(world_name, state):
    state = dict(state)
    state["fenra_version"] = FENRA_VERSION
    ensure_world_dir(world_name)
    with open(world_state_path(world_name), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ------------------------------------------------------------------ voices --

def voices_root_dir(world_name):
    return os.path.join(world_dir(world_name), "voices")


def voice_dir(world_name, voice_name):
    return os.path.join(voices_root_dir(world_name), voice_name)


def ensure_voice_dir(world_name, voice_name):
    path = voice_dir(world_name, voice_name)
    os.makedirs(path, exist_ok=True)
    return path


def voice_state_path(world_name, voice_name):
    return os.path.join(voice_dir(world_name, voice_name), VOICE_STATE_FILENAME)


def list_voices(world_name):
    root = voices_root_dir(world_name)
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and os.path.exists(voice_state_path(world_name, d))
    )


def default_voice_state():
    return {
        "model": DEFAULT_MODEL,
        "identity": "",
        "context": "",
        "currency": 10.0,
    }


def load_voice_state(world_name, voice_name):
    state = default_voice_state()
    path = voice_state_path(world_name, voice_name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                state.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return state


def save_voice_state(world_name, voice_name, state):
    state = dict(state)
    state["fenra_version"] = FENRA_VERSION
    ensure_voice_dir(world_name, voice_name)
    with open(voice_state_path(world_name, voice_name), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def delete_voice(world_name, voice_name):
    import shutil
    path = voice_dir(world_name, voice_name)
    if os.path.isdir(path):
        shutil.rmtree(path)


def append_to_context(world_name, voice_name, line):
    """The one and only way a voice's context grows - a plain string
    append, same format for a voice's own thought or an incoming one
    from a fellow group member (see the module docstring). Re-reads
    from disk immediately before appending rather than trusting an
    in-memory copy, so a concurrent write (Teddy editing Context in the
    GUI at the same moment) can't get silently clobbered."""
    state = load_voice_state(world_name, voice_name)
    existing = state.get("context", "")
    state["context"] = (existing + ("\n" if existing else "") + line)
    save_voice_state(world_name, voice_name, state)


# ------------------------------------------------------------------ groups --

def groups_root_dir(world_name):
    return os.path.join(world_dir(world_name), "groups")


def ensure_groups_root_dir(world_name):
    path = groups_root_dir(world_name)
    os.makedirs(path, exist_ok=True)
    return path


def group_path(world_name, name):
    name = sanitize_name(name)
    if not name or not _GROUP_NAME_RE.match(name):
        raise ValueError(
            "group names may only contain letters, numbers, underscores, and hyphens "
            f"(spaces and apostrophes get stripped automatically) - got '{name}'"
        )
    return os.path.join(groups_root_dir(world_name), f"{name}.json")


def list_groups(world_name):
    root = groups_root_dir(world_name)
    if not os.path.isdir(root):
        return []
    return sorted(
        f[: -len(".json")] for f in os.listdir(root)
        if f.endswith(".json") and os.path.isfile(os.path.join(root, f))
    )


def default_group_state(name):
    return {"name": name, "members": []}


def load_group_state(world_name, name):
    path = group_path(world_name, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_group_state(world_name, name, state):
    ensure_groups_root_dir(world_name)
    with open(group_path(world_name, name), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def delete_group(world_name, name):
    path = group_path(world_name, name)
    if os.path.exists(path):
        os.remove(path)


def groups_containing(world_name, voice_name):
    """Every group (sanitized name) that currently lists voice_name as
    a member - used both by the tick loop (who does a thought reach)
    and the Groups tab (member add/remove)."""
    out = []
    for name in list_groups(world_name):
        state = load_group_state(world_name, name)
        if state and voice_name in state.get("members", []):
            out.append(name)
    return out


def build_hud(world_name, voice_name):
    """The last thing in a voice's prompt (see module docstring) -
    computed fresh every tick, never written to state.json. Own
    name/model, own groups, every group in the world, who's currently
    seen (shares a group), who exists but isn't seen, then the voice's
    own identity line as the literal last line."""
    state = load_voice_state(world_name, voice_name)
    own_groups = groups_containing(world_name, voice_name)
    all_groups = list_groups(world_name)

    seen = set()
    for gname in own_groups:
        gstate = load_group_state(world_name, gname) or {}
        seen.update(gstate.get("members", []))
    seen.discard(voice_name)

    unseen = [v for v in list_voices(world_name) if v != voice_name and v not in seen]

    # Everyone's balance, not just your own (Teddy's call, 2026-09-10) -
    # full transparency rather than a private number, deliberately with
    # no goal attached. Sorted by balance so it reads as a standing.
    balances = []
    for v in list_voices(world_name):
        v_state = state if v == voice_name else load_voice_state(world_name, v)
        balances.append((v, v_state.get("currency", 0.0)))
    balances.sort(key=lambda pair: (-pair[1], pair[0]))
    currency_line = "Currency levels (everyone): " + ", ".join(
        f"{v}: ${amt:.2f}" for v, amt in balances
    )

    lines = [
        "Everything above this line is your thoughts. Everything below is your HUD.",
        f"Name: {voice_name}",
        f"Model: {state.get('model', DEFAULT_MODEL)}",
        f"Your groups: {', '.join(own_groups) if own_groups else 'none'}",
        f"All groups in this world: {', '.join(all_groups) if all_groups else 'none'}",
        f"Voices you can see: {', '.join(sorted(seen)) if seen else 'none'}",
        f"Voices that exist but you cannot see: {', '.join(unseen) if unseen else 'none'}",
        currency_line,
        "You can call functions by writing ⟦function_name(args)⟧ in your "
        "response - try ⟦functions()⟧ to see everything available to you.",
        state.get("identity", ""),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------- functions --

FUNCTION_CALL_RE = re.compile(r"⟦\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*⟧", re.DOTALL)


def _parse_target_and_rest(args_text):
    """Every current function takes "target|rest" - split once on the
    first '|', both sides stripped. Raises if the '|' is missing."""
    if "|" not in args_text:
        raise ValueError("expected 'target|...' - got no '|' separator")
    target, rest = args_text.split("|", 1)
    return target.strip(), rest.strip()


def fn_send_message(world_name, caller_name, args_text):
    """A direct message to one specific voice - the Slack-DM equivalent.
    Delivered straight into the target's real, persisted context via the
    same append_to_context every group broadcast already uses, just
    addressed to one voice and wrapped in an explicit flag so it reads
    as a message rather than ordinary group chatter (Teddy's call,
    2026-09-09 - not a separate transitory mechanism)."""
    target, text = _parse_target_and_rest(args_text)
    if target not in list_voices(world_name):
        raise ValueError(f"'{target}' isn't a voice in this world")
    if target == caller_name:
        raise ValueError("you can't send_message yourself")
    if not text:
        raise ValueError("no message text given")
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = (
        f"***You received the following message from {caller_name} at "
        f"{timestamp}*** {text} ***End Message from {caller_name}***"
    )
    append_to_context(world_name, target, line)
    return f"message sent to {target}"


def fn_give_currency(world_name, caller_name, args_text):
    """Real transfer between two voices' own stored currency balance."""
    target, amount_text = _parse_target_and_rest(args_text)
    if target not in list_voices(world_name):
        raise ValueError(f"'{target}' isn't a voice in this world")
    if target == caller_name:
        raise ValueError("you can't give_currency to yourself")
    # The HUD shows currency as "$10.00" - naturally, a voice copies that
    # formatting back when giving an amount (real case: Amanda wrote
    # "$2.00", 2026-09-09). Strip a leading $ and thousands-separator
    # commas so that still works instead of erroring.
    cleaned = amount_text.strip().lstrip("$").replace(",", "")
    try:
        amount = float(cleaned)
    except ValueError:
        raise ValueError(f"'{amount_text}' isn't a number")
    if amount <= 0:
        raise ValueError("amount must be positive")

    caller_state = load_voice_state(world_name, caller_name)
    balance = caller_state.get("currency", 0.0)
    if amount > balance:
        raise ValueError(f"you only have ${balance:.2f}, can't send ${amount:.2f}")
    caller_state["currency"] = balance - amount
    save_voice_state(world_name, caller_name, caller_state)

    target_state = load_voice_state(world_name, target)
    target_state["currency"] = target_state.get("currency", 0.0) + amount
    save_voice_state(world_name, target, target_state)

    return f"sent ${amount:.2f} to {target}"


def fn_functions(world_name, caller_name, args_text):
    """Lists the registry, optionally filtered by a substring in the
    name or description."""
    query = args_text.strip().lower() if args_text else None
    lines = []
    for name, meta in FUNCTION_REGISTRY.items():
        desc = meta["description"]
        if query and query not in name.lower() and query not in desc.lower():
            continue
        lines.append(f"{name}({meta['params']}): {desc}")
    return "\n".join(lines) if lines else "no matching functions"


FUNCTION_REGISTRY = {
    "send_message": {
        "fn": fn_send_message,
        "params": "target|text",
        "description": "Send a direct message to one specific voice - delivered into their context.",
    },
    "give_currency": {
        "fn": fn_give_currency,
        "params": "target|amount",
        "description": "Give some of your own currency to another voice.",
    },
    "functions": {
        "fn": fn_functions,
        "params": "[search term]",
        "description": "List everything you can call, optionally filtered by a search term.",
    },
}


def run_function_calls(world_name, caller_name, response_text):
    """Scans response_text for every ⟦function_name(args)⟧ call and runs
    each one for real. Returns a (full_text, masked_text) pair:

    - full_text: response_text with a ⟦RESULT: ...⟧ line appended per
      call - what the caller's own context gets (they made the call,
      they see what it actually did).
    - masked_text: response_text with each call replaced by a plain
      "(*caller called name*)" notice - no arguments, no result, ever -
      what gets broadcast to everyone else in a shared group (Teddy's
      call, 2026-09-10: calling a function shouldn't be any more visible
      to bystanders than a real action is - they can see *that* it
      happened, not the details, unless the caller chooses to say so in
      their own words).

    No calls found -> both entries are response_text unchanged."""
    matches = list(FUNCTION_CALL_RE.finditer(response_text))
    if not matches:
        return response_text, response_text

    result_lines = []
    for match in matches:
        name, args_text = match.group(1), match.group(2)
        meta = FUNCTION_REGISTRY.get(name)
        if not meta:
            result_lines.append(f"⟦RESULT: {name} -> error: unknown function '{name}'⟧")
            continue
        try:
            result = meta["fn"](world_name, caller_name, args_text)
            result_lines.append(f"⟦RESULT: {name} -> ok: {result}⟧")
        except Exception as exc:
            result_lines.append(f"⟦RESULT: {name} -> error: {exc}⟧")

    full_text = response_text + "\n" + "\n".join(result_lines)
    masked_text = FUNCTION_CALL_RE.sub(
        lambda m: f"(*{caller_name} called {m.group(1)}*)", response_text
    )
    return full_text, masked_text


# ------------------------------------------------------------------ model --

def call_ollama(host, model, prompt):
    # No fixed timeout, matching fenras-aletheosis's own REQUEST_TIMEOUT=None
    # (its comment applies here too, unchanged): some models are legitimately
    # slow, and a client-side timeout doesn't cancel server-side generation -
    # it just abandons the connection while the server keeps working anyway,
    # which can pile up rather than help.
    resp = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=None,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def list_ollama_models(host):
    try:
        resp = requests.get(f"{host}/api/tags", timeout=10)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))
    except requests.RequestException:
        return []


# ---------------------------------------------------------------- the app --

class FenraApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Fenra - worlds-rebuild v{FENRA_VERSION}")
        self.root.geometry("1000x650")

        self.world_name = None
        self.world_voices = []          # this world's round-robin order
        self.voice_rotation_index = 0
        self.running = False
        self.loop_thread = None

        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        self.interval_var = tk.StringVar(value=str(DEFAULT_INTERVAL_SEC))
        self.status_var = tk.StringVar(value="Idle")
        self.world_var = tk.StringVar(value="")

        self._current_voice_names = []   # listbox-index -> voice name
        self.displayed_voice = None
        self._current_group_names = []   # listbox-index -> group name
        self.displayed_group = None

        self._build_menu()
        self._build_ui()
        self._startup_world()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------ startup --

    def _startup_world(self):
        worlds = list_worlds()
        name = worlds[0] if worlds else DEFAULT_WORLD_NAME
        self._load_world(name)

    def _on_close(self):
        self.running = False
        if self.world_name:
            self._save_world_controls()
        self.root.destroy()

    # ---------------------------------------------------------------- UI --

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        worlds_menu = tk.Menu(file_menu, tearoff=0)
        worlds_menu.add_command(label="New world...", command=self.new_world)
        worlds_menu.add_command(label="Rename current world...", command=self.rename_world)
        worlds_menu.add_separator()
        self._worlds_menu = worlds_menu
        self._rebuild_worlds_menu()
        file_menu.add_cascade(label="Worlds", menu=worlds_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def _rebuild_worlds_menu(self):
        # Keep "New world.../Rename.../separator" (first 3 entries), drop
        # everything after, re-list every world fresh.
        self._worlds_menu.delete(3, "end")
        for name in list_worlds():
            self._worlds_menu.add_command(label=name, command=lambda n=name: self._load_world(n))

    def _build_ui(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=6, pady=4)
        ttk.Label(toolbar, text="World:").pack(side="left")
        ttk.Label(toolbar, textvariable=self.world_var, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(2, 14))
        ttk.Label(toolbar, text="Host:").pack(side="left")
        ttk.Entry(toolbar, textvariable=self.host_var, width=24).pack(side="left", padx=(2, 10))
        ttk.Label(toolbar, text="Interval (s):").pack(side="left")
        ttk.Entry(toolbar, textvariable=self.interval_var, width=5).pack(side="left", padx=(2, 10))
        self.start_stop_btn = ttk.Button(toolbar, text="Start", command=self.toggle_loop)
        self.start_stop_btn.pack(side="left", padx=(0, 10))
        ttk.Label(toolbar, textvariable=self.status_var, foreground="#666").pack(side="left")

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        self.voices_tab = ttk.Frame(notebook)
        self.groups_tab = ttk.Frame(notebook)
        self.currency_tab = ttk.Frame(notebook)
        notebook.add(self.voices_tab, text="Voices")
        notebook.add(self.groups_tab, text="Groups")
        notebook.add(self.currency_tab, text="Currency")

        self._build_voices_tab()
        self._build_groups_tab()
        self._build_currency_tab()

    # ----------------------------------------------------------- Voices tab --

    def _build_voices_tab(self):
        frame = self.voices_tab

        top_bar = ttk.Frame(frame)
        top_bar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(top_bar, text="New voice...", command=self.new_voice).pack(side="left", padx=2)
        ttk.Button(top_bar, text="Delete voice", command=self.delete_voice).pack(side="left", padx=2)
        ttk.Button(top_bar, text="Save voice", command=self.save_voice).pack(side="left", padx=2)

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(paned, width=180)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=4)

        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.voices_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False)
        scrollbar.config(command=self.voices_listbox.yview)
        self.voices_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.voices_listbox.bind("<<ListboxSelect>>", self._on_voice_select)

        params_row = ttk.Frame(right)
        params_row.pack(fill="x", pady=(0, 4))
        ttk.Label(params_row, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_combo = ttk.Combobox(params_row, textvariable=self.model_var, width=24, state="normal")
        self.model_combo.pack(side="left", padx=(2, 4))
        ttk.Button(params_row, text="↻", width=3, command=self.refresh_models).pack(side="left")

        ttk.Label(right, text="Identity (last line of the HUD, every cycle):").pack(anchor="w", padx=2)
        self.identity_box = scrolledtext.ScrolledText(right, wrap="word", height=6)
        self.identity_box.pack(fill="x", padx=2, pady=(0, 4))

        ttk.Label(right, text="Context (grows automatically - fully editable):").pack(anchor="w", padx=2)
        self.context_box = scrolledtext.ScrolledText(right, wrap="word")
        self.context_box.pack(fill="both", expand=True, padx=2, pady=(0, 2))

    def _populate_voices_list(self):
        self._current_voice_names = list_voices(self.world_name)
        self.voices_listbox.delete(0, "end")
        for name in self._current_voice_names:
            self.voices_listbox.insert("end", name)

    def _on_voice_select(self, event):
        selection = self.voices_listbox.curselection()
        if not selection:
            return
        if self.displayed_voice:
            self._save_voice_snapshot(self.displayed_voice)
        name = self._current_voice_names[selection[0]]
        self._load_voice(name)

    def _load_voice(self, name):
        state = load_voice_state(self.world_name, name)
        self.displayed_voice = name
        self.model_var.set(state.get("model", DEFAULT_MODEL))
        self.identity_box.delete("1.0", "end")
        self.identity_box.insert("end", state.get("identity", ""))
        self.context_box.delete("1.0", "end")
        self.context_box.insert("end", state.get("context", ""))

    def _save_voice_snapshot(self, name):
        state = {
            "model": self.model_var.get(),
            "identity": self.identity_box.get("1.0", "end-1c"),
            "context": self.context_box.get("1.0", "end-1c"),
        }
        save_voice_state(self.world_name, name, state)

    def save_voice(self):
        if not self.displayed_voice:
            return
        self._save_voice_snapshot(self.displayed_voice)
        self.status_var.set(f"Saved '{self.displayed_voice}'")

    def new_voice(self):
        name = simpledialog.askstring("New Voice", "Voice name:", parent=self.root)
        if not name:
            return
        name = sanitize_name(name)
        if not name:
            return
        if name in list_voices(self.world_name):
            messagebox.showerror("Fenra", f"a voice named '{name}' already exists.")
            return
        save_voice_state(self.world_name, name, default_voice_state())
        self.world_voices.append(name)
        self._save_world_controls()
        self._populate_voices_list()
        self._refresh_group_member_candidates()

    def delete_voice(self):
        if not self.displayed_voice:
            return
        name = self.displayed_voice
        if not messagebox.askyesno("Fenra", f"Delete voice '{name}'? This can't be undone."):
            return
        delete_voice(self.world_name, name)
        if name in self.world_voices:
            self.world_voices.remove(name)
        # Also drop it from every group's membership - a deleted voice
        # can't stay listed as a member of anything.
        for gname in list_groups(self.world_name):
            gstate = load_group_state(self.world_name, gname)
            if gstate and name in gstate.get("members", []):
                gstate["members"].remove(name)
                save_group_state(self.world_name, gname, gstate)
        self._save_world_controls()
        self.displayed_voice = None
        self._populate_voices_list()
        self._refresh_group_member_candidates()
        if self.displayed_group:
            self._load_group(self.displayed_group)

    def refresh_models(self):
        models = list_ollama_models(self.host_var.get())
        self.model_combo["values"] = models
        if models:
            self.status_var.set(f"{len(models)} model(s) available")
        else:
            self.status_var.set("Could not reach Ollama host")

    # ----------------------------------------------------------- Groups tab --

    def _build_groups_tab(self):
        frame = self.groups_tab

        top_bar = ttk.Frame(frame)
        top_bar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(top_bar, text="New group...", command=self.new_group).pack(side="left", padx=2)
        ttk.Button(top_bar, text="Delete group", command=self.delete_group).pack(side="left", padx=2)
        ttk.Button(top_bar, text="Rename group", command=self.rename_group).pack(side="left", padx=2)

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(paned, width=180)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.groups_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, exportselection=False)
        scrollbar.config(command=self.groups_listbox.yview)
        self.groups_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.groups_listbox.bind("<<ListboxSelect>>", self._on_group_select)

        name_row = ttk.Frame(right)
        name_row.pack(fill="x", padx=2, pady=(0, 6))
        ttk.Label(name_row, text="Name:", width=12).pack(side="left")
        self.group_name_var = tk.StringVar(value="")
        ttk.Label(name_row, textvariable=self.group_name_var, font=("Segoe UI", 9, "bold")).pack(side="left")

        members_frame = ttk.LabelFrame(right, text="Members")
        members_frame.pack(fill="both", expand=True, padx=2, pady=(0, 4))
        self.group_members_listbox = tk.Listbox(members_frame, selectmode="extended", exportselection=False)
        self.group_members_listbox.pack(fill="both", expand=True, padx=4, pady=(4, 2))
        ttk.Button(members_frame, text="Remove selected", command=self.remove_group_members).pack(
            anchor="e", padx=4, pady=(0, 4)
        )

        add_row = ttk.Frame(right)
        add_row.pack(fill="x", padx=2, pady=(0, 4))
        ttk.Label(add_row, text="Add voice:").pack(side="left")
        self.group_add_voice_var = tk.StringVar(value="")
        self.group_add_voice_combo = ttk.Combobox(
            add_row, textvariable=self.group_add_voice_var, width=20, state="readonly"
        )
        self.group_add_voice_combo.pack(side="left", padx=(4, 4))
        ttk.Button(add_row, text="Add", command=self.add_group_member).pack(side="left")

    def _populate_groups_list(self):
        self._current_group_names = list_groups(self.world_name)
        self.groups_listbox.delete(0, "end")
        for name in self._current_group_names:
            self.groups_listbox.insert("end", name)

    def _on_group_select(self, event):
        selection = self.groups_listbox.curselection()
        if not selection:
            return
        self._load_group(self._current_group_names[selection[0]])

    def _load_group(self, name):
        state = load_group_state(self.world_name, name) or default_group_state(name)
        self.displayed_group = name
        self.group_name_var.set(state.get("name", name))
        self.group_members_listbox.delete(0, "end")
        for voice in state.get("members", []):
            self.group_members_listbox.insert("end", voice)
        self._refresh_group_member_candidates()

    def _refresh_group_member_candidates(self):
        """The 'Add voice' combobox - every world voice not already a
        member of the currently displayed group."""
        if not self.displayed_group:
            self.group_add_voice_combo["values"] = []
            return
        state = load_group_state(self.world_name, self.displayed_group) or {}
        members = set(state.get("members", []))
        candidates = [v for v in list_voices(self.world_name) if v not in members]
        self.group_add_voice_combo["values"] = candidates
        if candidates:
            self.group_add_voice_var.set(candidates[0])
        else:
            self.group_add_voice_var.set("")

    def new_group(self):
        name = simpledialog.askstring("New Group", "Group name:", parent=self.root)
        if not name:
            return
        name = sanitize_name(name)
        if not name:
            return
        if name in list_groups(self.world_name):
            messagebox.showerror("Fenra", f"a group named '{name}' already exists.")
            return
        save_group_state(self.world_name, name, default_group_state(name))
        self._populate_groups_list()

    def rename_group(self):
        if not self.displayed_group:
            return
        old_name = self.displayed_group
        new_name = simpledialog.askstring("Rename Group", "New name:", initialvalue=old_name, parent=self.root)
        if not new_name:
            return
        new_name = sanitize_name(new_name)
        if not new_name or new_name == old_name:
            return
        if new_name in list_groups(self.world_name):
            messagebox.showerror("Fenra", f"a group named '{new_name}' already exists.")
            return
        state = load_group_state(self.world_name, old_name) or default_group_state(old_name)
        state["name"] = new_name
        save_group_state(self.world_name, new_name, state)
        delete_group(self.world_name, old_name)
        self.displayed_group = new_name
        self._populate_groups_list()

    def delete_group(self):
        if not self.displayed_group:
            return
        name = self.displayed_group
        if not messagebox.askyesno("Fenra", f"Delete group '{name}'? This can't be undone."):
            return
        delete_group(self.world_name, name)
        self.displayed_group = None
        self._populate_groups_list()
        self.group_name_var.set("")
        self.group_members_listbox.delete(0, "end")

    def add_group_member(self):
        if not self.displayed_group:
            return
        voice = self.group_add_voice_var.get()
        if not voice:
            return
        state = load_group_state(self.world_name, self.displayed_group) or default_group_state(self.displayed_group)
        if voice not in state["members"]:
            state["members"].append(voice)
            save_group_state(self.world_name, self.displayed_group, state)
        self._load_group(self.displayed_group)

    def remove_group_members(self):
        if not self.displayed_group:
            return
        selection = self.group_members_listbox.curselection()
        if not selection:
            return
        to_remove = {self.group_members_listbox.get(i) for i in selection}
        state = load_group_state(self.world_name, self.displayed_group) or default_group_state(self.displayed_group)
        state["members"] = [v for v in state.get("members", []) if v not in to_remove]
        save_group_state(self.world_name, self.displayed_group, state)
        self._load_group(self.displayed_group)

    # --------------------------------------------------------- Currency tab --

    def _build_currency_tab(self):
        frame = self.currency_tab

        top_bar = ttk.Frame(frame)
        top_bar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(top_bar, text="Refresh now", command=self._populate_currency_list).pack(side="left", padx=2)
        ttk.Label(
            top_bar,
            text="Read-only - every voice's balance, highest first. Auto-refreshes every 10s.",
            foreground="#666",
        ).pack(side="left", padx=(8, 0))

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill="both", expand=True, padx=6, pady=6)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.currency_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.currency_listbox.yview)
        self.currency_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._schedule_currency_refresh()

    def _populate_currency_list(self):
        if not self.world_name:
            return
        balances = []
        for name in list_voices(self.world_name):
            state = load_voice_state(self.world_name, name)
            balances.append((name, state.get("currency", 0.0)))
        balances.sort(key=lambda pair: (-pair[1], pair[0]))

        self.currency_listbox.delete(0, "end")
        for name, amount in balances:
            self.currency_listbox.insert("end", f"{name}: ${amount:.2f}")

    def _schedule_currency_refresh(self):
        self._populate_currency_list()
        self.root.after(CURRENCY_REFRESH_MS, self._schedule_currency_refresh)

    # --------------------------------------------------------------- worlds --

    def _load_world(self, name):
        if self.running:
            self.toggle_loop()
        if self.displayed_voice:
            self._save_voice_snapshot(self.displayed_voice)

        ensure_world_dir(name)
        state = load_world_state(name)
        self.world_name = name
        self.world_var.set(name)
        self.host_var.set(state.get("host", DEFAULT_HOST))
        self.interval_var.set(str(state.get("interval", DEFAULT_INTERVAL_SEC)))
        self.world_voices = state.get("voices", [])
        self.voice_rotation_index = state.get("voice_rotation_index", 0)

        self.displayed_voice = None
        self.displayed_group = None
        self._populate_voices_list()
        self._populate_groups_list()
        self.group_name_var.set("")
        self.group_members_listbox.delete(0, "end")
        self.identity_box.delete("1.0", "end")
        self.context_box.delete("1.0", "end")
        self.status_var.set("Idle")
        self._rebuild_worlds_menu()

    def new_world(self):
        name = simpledialog.askstring("New World", "World name:", parent=self.root)
        if not name:
            return
        name = sanitize_name(name)
        if not name:
            return
        if name in list_worlds():
            if messagebox.askyesno("Fenra", f"World '{name}' already exists. Load it instead?"):
                self._load_world(name)
            return
        ensure_world_dir(name)
        save_world_state(name, default_world_state())
        self._load_world(name)

    def rename_world(self):
        if not self.world_name:
            return
        old_name = self.world_name
        new_name = simpledialog.askstring("Rename World", "New name:", initialvalue=old_name, parent=self.root)
        if not new_name:
            return
        new_name = sanitize_name(new_name)
        if not new_name or new_name == old_name:
            return
        if new_name in list_worlds():
            messagebox.showerror("Fenra", f"a world named '{new_name}' already exists.")
            return
        # Real bug, caught by the live smoke test: _load_world's own
        # save-before-switch step (correct for an ordinary world switch,
        # where the world being left genuinely still exists on disk)
        # writes into self.world_name, which is still the OLD name at
        # that point since _load_world only updates it partway through.
        # For a rename specifically, the old directory has just been
        # physically moved away - writing to it afterward resurrects a
        # stale, empty duplicate under the old name. Clearing the
        # displayed selection first means _load_world has nothing to
        # save before the switch, since there's no "old world" left to
        # save it into - this world didn't go away, it just changed name.
        self.displayed_voice = None
        self.displayed_group = None
        os.rename(world_dir(old_name), world_dir(new_name))
        self._load_world(new_name)

    def _save_world_controls(self):
        if not self.world_name:
            return
        state = {
            "host": self.host_var.get(),
            "interval": self.interval_var.get(),
            "model_default": DEFAULT_MODEL,
            "voices": self.world_voices,
            "voice_rotation_index": self.voice_rotation_index,
        }
        save_world_state(self.world_name, state)

    # ----------------------------------------------------------- the loop --

    def toggle_loop(self):
        if self.running:
            self.running = False
            self.start_stop_btn.config(text="Start")
            self.status_var.set("Stopping...")
        else:
            if not self.world_voices:
                messagebox.showinfo("Fenra", "This world has no voices yet - add one on the Voices tab first.")
                return
            self.running = True
            self.start_stop_btn.config(text="Stop")
            self.status_var.set("Running")
            self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
            self.loop_thread.start()

    def _run_loop(self):
        while self.running:
            try:
                self._tick()
            except Exception as exc:  # keep the loop alive on a transient error
                self.root.after(0, self.status_var.set, f"Error: {exc}")
            try:
                interval = float(self.interval_var.get())
            except ValueError:
                interval = DEFAULT_INTERVAL_SEC
            for _ in range(int(interval * 10)):
                if not self.running:
                    break
                time.sleep(0.1)
        self.root.after(0, self.status_var.set, "Idle")

    def _tick(self):
        if not self.world_voices:
            return
        # If the currently-displayed voice is about to run, its in-flight
        # widget edits are the authoritative copy - persist them first so
        # a same-voice tick doesn't clobber an unsaved edit.
        if self.displayed_voice:
            self.root.after(0, self._save_voice_snapshot, self.displayed_voice)

        index = self.voice_rotation_index % len(self.world_voices)
        active_voice = self.world_voices[index]
        self.voice_rotation_index = (index + 1) % len(self.world_voices)
        self.root.after(0, self._save_world_controls)

        state = load_voice_state(self.world_name, active_voice)
        model = state.get("model", DEFAULT_MODEL)
        prompt = f"{state.get('context', '')}\n\n{build_hud(self.world_name, active_voice)}"
        try:
            response = call_ollama(self.host_var.get(), model, prompt)
        except requests.RequestException as exc:
            self.root.after(0, self.status_var.set, f"Error calling {model}: {exc}")
            return
        response = response.strip()
        if not response:
            return
        full_response, masked_response = run_function_calls(self.world_name, active_voice, response)

        timestamp = datetime.now().isoformat(timespec="seconds")
        full_line = f"[{timestamp}] {active_voice}: {full_response}"
        masked_line = f"[{timestamp}] {active_voice}: {masked_response}"

        # The speaker's own record - the real thing, calls and results
        # both. Everyone else sees the masked version (see
        # run_function_calls) - that a call happened, never its
        # arguments or result.
        append_to_context(self.world_name, active_voice, full_line)

        # Every OTHER member of every group the speaker belongs to - one
        # append per listener max, even if they share more than one
        # group with the speaker.
        already_notified = {active_voice}
        for group_name in groups_containing(self.world_name, active_voice):
            gstate = load_group_state(self.world_name, group_name) or {}
            for member in gstate.get("members", []):
                if member in already_notified:
                    continue
                already_notified.add(member)
                append_to_context(self.world_name, member, masked_line)

        if active_voice == self.displayed_voice:
            self.root.after(0, self._load_voice, active_voice)
        self.root.after(0, self.status_var.set, f"Running ('{active_voice}' spoke)")


def main():
    root = tk.Tk()
    app = FenraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
