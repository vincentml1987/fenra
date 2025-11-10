from __future__ import annotations

from typing import Dict, List
import importlib


def apply_and_persist_pdv_adjustments(adjs: List[Dict], *, scale: float = 1.0) -> dict[str, float]:
    """
    Normalize PDV adjustments and forward them to Conductor's additive updater.
    No upper clamp; floor at zero is handled by Conductor.
    """
    # Import lazily to avoid circular imports if called from UI/Discord.
    conductor = importlib.import_module("conductor")

    norm: list[dict] = []
    for a in adjs or []:
        name = a.get("name") or a.get("pdv")
        if not name:
            continue
        raw_delta = a.get("delta")
        if raw_delta is None:
            raw_delta = a.get("delta_pct")
        if raw_delta is None:
            continue
        try:
            delta = float(raw_delta)
        except Exception:
            continue
        norm.append({"name": name, "delta": delta})

    if not norm:
        return {}

    # Conductor applies +delta * scale, floors at 0, persists pdvs.json and history.
    return conductor.apply_pdv_adjustments(norm, scale=scale)
