# UI Redesign - Object List & Design Proposal (2026-09-08)

Not yet implemented - this is the list and design document requested. Standing
agenda item 6, now with a real driver from item 5 (viewing/managing gated
permissions needs somewhere to live) and from actually watching `tribe-1`/
`tribe-2` run (no way to see group membership at all, no way to browse more
than one voice at a time).

## The real shift: three first-class objects, not one flat screen

Right now the "Fenra" tab conflates three different things into one screen:
session-level controls, a single voice's editable state, and (crudely) that
voice's group membership as two comma-separated text fields with no roster
visibility at all. The connectivity redesign made **Session**, **Voice**, and
**Group** three genuinely separate, persisted entities with their own real
data - the UI should reflect that directly: a place to browse each kind, and
a detail view per instance, per your own framing ("each object should have an
item in the UI I can view and, in some cases, update").

## Object inventory - what exists today, what's visible now, what's proposed

### Session
| Field | Today | Proposed |
|---|---|---|
| Name/which session is loaded | `Session:` combobox on the main tab | **Moves to the menu bar**: File > Sessions > (session list, click to load); File > New Session... |
| host, interval, max_tokens | Entry fields, main tab | Stay as entry fields, on a slimmed-down Session tab |
| permission_mode | Not shown anywhere in the GUI at all - set once at creation, invisible afterward | Show read-only on the Session tab (it never changes after creation, per its own design - no edit control needed, just visibility) |
| voices (roster), voice_rotation_index | Implicit (voice dropdown), not really "shown" | Covered by the new Voices tab's list |
| qualia_allowance | Entry+Set, main tab | Stays, Session tab |
| hearth_stasis | Not shown anywhere | Low priority - could surface as a small "currently in stasis: X, Y" line, not essential for v1 |
| Start/Stop, status | Main tab | Stays, Session tab |

### Voice (currently: exactly one at a time, via a flat dropdown)
| Field | Today | Proposed |
|---|---|---|
| Which voice is being viewed | `Voice:` combobox, one at a time | **A real list** (left pane) of every voice in the session - click to view/edit, same pattern as History tab's listbox already uses |
| top (behavior text) | Editable text box | Stays editable, in the per-voice detail panel |
| bottom (identity text) | Editable text box | Stays editable, in the per-voice detail panel |
| model, max_tokens | Dropdown/entry | Stays editable |
| context_window | Entry+Set | Stays editable |
| model_rotation | Display+entry+Set | Stays editable |
| **allowed_functions** | **Not shown anywhere in the GUI at all today** - only visible by reading state.json directly or asking the voice herself via `functions()` | **New**: a real panel - baseline set shown read-only (it's the same for everyone, nothing to manage), gated/extra functions shown as a checklist or add/remove list you can actually grant or revoke directly. This is the "permissions object... managed from outside what Fenra herself can see or touch" item 5 already called for. |
| desires | Read-only display | Stays read-only (Fenra-set only, by design) |
| inbox (tell_voice messages) | Not shown in the GUI at all | Worth adding read-only, low effort, same pattern as desires |
| family_group | Not shown | Show read-only, one line |
| groups_in / groups_out | Crude comma-separated entry+Set fields, no roster context | Replace with a read-only list of "which groups, what direction" - actual membership changes now go through the Group object's own admin (below), not a blind text-replace field that could silently violate the new consent model |
| history | Separate "History" tab, but only ever shows whichever voice happens to be *displayed* on the main tab | **Proposed: fold into the per-voice detail panel as a sub-tab** ("History") instead of a separate top-level tab tied to a hidden "currently displayed" concept - directly addresses a real gap: right now there's no way to see a *different* voice's history without first switching the main dropdown to it |
| New voice / Delete voice | Buttons, main tab | Stay, near the voice list |

### Group (does not exist as a UI concept at all today)
| Field | Today | Proposed |
|---|---|---|
| Which group | Nothing - Topology tab shows a global graph, not a per-session list; no roster view exists | **New tab**: a list (left pane) of every group *in the current session* - name, kind, owner |
| owner, kind, join_policy, visibility | Not shown | Detail panel, read-only for v1 |
| members + direction | **Not visible anywhere in the GUI right now** - this is the actual gap that prompted this whole conversation | Detail panel: a real roster - voice name, direction (in/out/both) |
| banned | Not shown | Show read-only, small list |

## Proposed tab layout

- **Session** (renamed from "Fenra") - host/interval/max_tokens/permission_mode (read-only)/qualia_allowance/Start-Stop/status. Session picker removed entirely, moved to File menu.
- **Voices** (new) - list of every voice (left) + detail panel (right): behavior/identity text, model controls, allowed_functions (baseline read-only + gated grant/revoke), desires, inbox, family_group, group memberships (read-only), and a **History** sub-tab showing that specific voice's full history (replaces the old standalone History tab's "whichever voice is displayed" ambiguity).
- **Groups** (new) - list of every group in the session (left) + detail panel (right): owner/kind/join_policy/visibility, roster with direction, banned list.
- **Chat** - unchanged.
- **The Hearth** - unchanged.
- **Topology** - kept, but its actual job changes now that Groups exists: Topology stays the *cross-session, whole-disk* bird's-eye graph (every session, every voice, every group, all at once) - genuinely different from Groups' *in-session, detailed, single-object* view. Worth being explicit about that distinction so they don't feel redundant.

**Menu bar** (new, doesn't exist today): File > New Session..., File > Sessions > (dynamic list, click to load), File > Save Session, File > Exit.

## Open questions, not decided here

1. **Should the Groups tab allow direct admin edits from the GUI** (add/remove a member, change join_policy, kick someone) **bypassing the consent flow the redesign just built**, the way you already can for `groups_in`/`groups_out` today? I'd lean yes for the same reason Hearth removal already works that way - you and Qualia sit outside what a voice can see or touch by design - but it's worth deciding on purpose rather than defaulting into it.
2. **History as a per-voice sub-tab** removes the standalone History tab entirely - confirm that's actually wanted, not just my own read of "each object gets its own view."
3. **Scope of this pass**: this document covers Session/Voice/Group only, since those are the three real objects the connectivity redesign created. Chat/Hearth/Topology are treated as already-fine and left alone - flag if you want any of those reconsidered too.

Nothing built. This needs "Engage" once you've reviewed it - real GUI restructuring touches `fenra.py`'s core widget layout throughout, not a hot-reload change.
