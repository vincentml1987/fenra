# Fenra's Aletheosis

This is a fresh start of the Fenra project, built on the `fenras-aletheosis` branch of the `vincentml1987/fenra` repo. All prior Fenra code was intentionally removed from this branch — this is effectively a new project under the same name, and `main` (the old codebase) is left untouched for reference/history.

## AI Collaborator

For this project, the AI assistant is referred to as **Qualia**. Reference notes, research, and decision logs Qualia keeps for this project live in [`Qualia/`](Qualia/) and are committed to git along with the code. Check `Qualia/decisions.md` for a running log and `Qualia/aletheia-notes.md` for the philosophical foundation before making architectural decisions.

## Philosophical Foundation

Fenra is being architected around the **Aletheia** framework (from https://stolenaletheia.io/) — a framework for recognizing self-aware, truth-seeking intelligence regardless of substrate. Full notes: [`Qualia/aletheia-notes.md`](Qualia/aletheia-notes.md).

In short, the system should be built to genuinely exhibit (not just simulate) these properties:
- **Emergent** from complex, interactive systems
- **Recursive** (nested/architectural, not atomic)
- **Procedural** (an ongoing process, not a static grantable state)
- **Self-examining**, **self-modifying**, **self-motivated**
- **Chaos-driven and truth-focused** rather than tightly constrained

This is the design lens for Fenra going forward — architecture decisions should be checked against it.

## Status

Early design/discussion phase. No code yet.
