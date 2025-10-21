from __future__ import annotations
from typing import Callable, Dict, Tuple, Any
import ast
import subprocess
import sys
import shutil
import re

# ---------------------------
# Function registry and API
# ---------------------------

_REGISTRY: Dict[str, tuple[Callable[..., str], str]] = {}


def register(name: str, description: str):
    """Decorator to register a callable Fenra function with a human-readable description."""

    def _wrap(func: Callable[..., str]):
        _REGISTRY[name] = (func, description)
        return func

    return _wrap


def _like_to_regex(pattern: str) -> re.Pattern:
    """Translate a % wildcard pattern to a case-insensitive regex."""
    escaped = re.escape(pattern)
    rx = ".*" if not escaped else escaped.replace(r"\%", ".*")
    return re.compile(rx, re.IGNORECASE)


def _literalize(node: ast.AST) -> Any:
    """Safely convert AST nodes for args/kwargs via literal evaluation."""
    return ast.literal_eval(node)


def _parse_call(expr: str) -> tuple[str, list[Any], dict[str, Any]]:
    """
    Parse 'fn_name(arg, kw=val, ...)' into (name, args, kwargs).
    Raises ValueError on any invalid or unsafe expression.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression: {e}") from e

    if not isinstance(tree.body, ast.Call) or not isinstance(tree.body.func, ast.Name):
        raise ValueError("expression is not a simple function call")

    fn_name = tree.body.func.id
    args = [_literalize(a) for a in tree.body.args]
    kwargs = {kw.arg: _literalize(kw.value) for kw in tree.body.keywords if kw.arg is not None}
    return fn_name, args, kwargs


def dispatch_expression(expr: str) -> tuple[str, bool, str]:
    """
    Dispatch a Fenra function expression found inside *~...~*.
    Returns (function_name_or_guess, found, result_string).
    - found=False when the name is not registered -> 'Function does not exist.'
    - found=True for executed or error-returning calls (errors are returned as strings).
    """
    guessed_name = expr.strip()
    try:
        name, args, kwargs = _parse_call(expr)
        guessed_name = name
    except Exception:
        return (guessed_name, False, "Function does not exist.")

    entry = _REGISTRY.get(name)
    if not entry:
        return (name, False, "Function does not exist.")

    func, _desc = entry
    try:
        res = func(*args, **kwargs)
        return (name, True, "" if res is None else str(res))
    except Exception as e:
        return (name, True, f"(error) {type(e).__name__}: {e}")


# --------------------------------
# Built-in/required Fenra functions
# --------------------------------


@register("list_functions", "List available Fenra functions; supports % wildcard on name/description.")
def list_functions(search: str = "") -> str:
    """
    Return a newline-separated list of 'name: description'.
    If search is provided, use % as wildcard and match name/description (case-insensitive).
    Always includes this function in the registry.
    """
    _REGISTRY.setdefault(
        "list_functions",
        (list_functions, "List available Fenra functions; supports % wildcard on name/description."),
    )

    items = []
    if search:
        rx = _like_to_regex(search)
        for name, (_fn, desc) in _REGISTRY.items():
            if rx.search(name) or rx.search(desc or ""):
                items.append(f"{name}: {desc}")
    else:
        for name in sorted(_REGISTRY):
            items.append(f"{name}: {_REGISTRY[name][1]}")
    return "\n".join(items) if items else "(no matching functions)"


@register("fenra_powershell", "Execute a PowerShell command string and return its output.")
def fenra_powershell(command: str) -> str:
    """
    Execute a PowerShell command and return its output as text.
    This is the ONLY path for running PowerShell from agent replies.
    """
    if not isinstance(command, str) or not command.strip():
        return "(no command)"

    exe = shutil.which("pwsh") or shutil.which("powershell")
    if exe is None:
        exe = "powershell.exe" if sys.platform.startswith("win") else "pwsh"

    args = [exe, "-NoProfile", "-NonInteractive", "-Command", command]
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        combined = out if not err else (out + ("\n" if out and err else "") + err)
        return combined.strip()
    except Exception as e:
        return f"(error) {type(e).__name__}: {e}"

