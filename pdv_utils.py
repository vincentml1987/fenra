import os
import time
import json
from typing import List, Dict

from config_loader import load_globals, load_pdvs, save_pdvs


def _clamp01(x: float) -> float:
    """Clamp *x* to the range [0.0, 1.0]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _ensure_dirs() -> None:
    """Ensure output directories for PDV logs exist."""
    os.makedirs("chatlogs", exist_ok=True)


def _pdv_values_map(pdvs_cfg: Dict[str, dict]) -> Dict[str, float]:
    """Return a map of PDV name -> float value from the config mapping."""
    return {name: float(cfg.get("value", 0.5)) for name, cfg in pdvs_cfg.items()}


def apply_and_persist_pdv_adjustments(adjs: List[dict]) -> Dict[str, float]:
    """Apply gamma-scaled PDV deltas and persist results.

    Parameters
    ----------
    adjs: list of dict
        Adjustment items describing the PDV name and delta.

    Returns
    -------
    dict
        Mapping of PDV name -> current value after applying adjustments.
    """
    globals_cfg = load_globals()
    try:
        gamma = float(globals_cfg.get("pdv_gamma", 2.0))
    except Exception:
        gamma = 2.0

    pdvs_cfg = load_pdvs()
    values = _pdv_values_map(pdvs_cfg)

    changed = False
    for item in adjs or []:
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue

        delta = None
        if "delta_pct" in item:
            try:
                delta = float(item["delta_pct"]) / 100.0
            except Exception:
                continue
        elif "delta" in item:
            try:
                delta = float(item["delta"])
            except Exception:
                continue
        if delta is None:
            continue

        x = float(values.get(name, 0.5))
        g = (4.0 * x * (1.0 - x)) ** gamma
        x2 = _clamp01(x + delta * g)

        if abs(x2 - x) > 1e-12:
            values[name] = x2
            if name not in pdvs_cfg:
                pdvs_cfg[name] = {"name": name, "description": "", "value": x2}
            else:
                pdvs_cfg[name]["value"] = x2
            changed = True

    if changed:
        save_pdvs(pdvs_cfg)
        _ensure_dirs()
        history_path = os.path.join("chatlogs", "pdv_history.jsonl")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "pdvs": values}, ensure_ascii=False) + "\n")
        live_path = os.path.join("chatlogs", "pdvs_live.json")
        with open(live_path, "w", encoding="utf-8") as f:
            json.dump(values, f)

    return values
