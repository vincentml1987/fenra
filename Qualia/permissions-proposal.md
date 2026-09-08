# Function Permissions Proposal - Review Draft (2026-09-08)

Not yet implemented. Standing agenda item 5 ("redesign how permissions work,
function-by-function") finally getting a real rule to apply, prompted by a
real observation from `tribe-1`'s first cycle: desires (and several other
purely self-directed things) shouldn't require permission at all.

## The rule, as refined in conversation

A function is **baseline** (always callable, regardless of `allowed_functions`
or `permission_mode`) if it only ever affects the calling voice's own state,
or if its own internal logic already provides the real safety check (see
`join_group` below). Everything else - anything that reaches into another
voice's state, a shared resource with real consequences, external network
access, or administrative power over something other voices depend on -
stays **gated**: must be explicitly granted via the existing
request/approve/grant pattern, never assumed.

## The bigger change this forces: no more inheritance at creation

Built into Step 2 of the connectivity redesign, now being reversed:
`fn_create_voice` currently snapshot-copies the parent's entire
`allowed_functions` list into the child. Teddy's catch: this means `seed`
handing out `create_voice` to itself once means every single descendant,
forever, ends up holding every extra power `seed` ever accumulated, whether
or not that specific child has any use for it - the opposite of deliberate,
function-by-function gating.

**Proposed fix:** a freshly created voice's `allowed_functions` starts empty,
same as before the redesign. It gets every *baseline* function for free
regardless (that's what baseline means now - not gated on `allowed_functions`
at all), but every *gated* function - `create_voice` included - has to be
requested and granted after the fact, same as any other voice. Whoever holds
`check_function_requests`/`approve_function_request`/etc. (`seed`, to start)
stays the actual keeper of gated permissions - nothing about *that* changes,
only that a child no longer walks away from its own birth already holding
whatever its parent happened to accumulate.

**Code impact once this is confirmed:** revert the `allowed_functions` copy
loop added in `fn_create_voice` (fenra_functions.py) back to leaving it as
`default_voice_state()`'s empty default. Everything else about `create_voice`
(family-group auto-join, model/context_window carrying over) stays as built.

## Full function list

### Baseline - self-only, no downstream effect on anyone else

| Function | Rationale |
|---|---|
| `functions` | Already global - discovery only. |
| `now` | Pure computation, no state at all. |
| `add_desire` | Own desire queue only. The case that started this conversation. |
| `set_context_window` | Own context window only. |
| `current_model` | Own state, read-only. |
| `set_model` | Own model choice only. |
| `add_to_rotation` | Own model rotation only. |
| `list_voices` | Already baseline (v0.16.15) - pure awareness. |
| `list_groups` | Already baseline (v0.16.15) - pure awareness, respects private opacity. |
| `tell_voice` | Already baseline (v0.16.15) - reaches another voice's inbox, but Teddy's own prior call: basic 1:1 contact shouldn't need permission. |
| `request_function_access` | Already global - asking is never restricted. |
| `request_group_join` | Already baseline (v0.16.15) - symmetric with the above. |
| `leave_group` | **Confirmed by Teddy** - "very much a self thing." Only ever removes your own membership. |
| `join_group` | **Confirmed by Teddy**, with a specific reason: the function's *own* logic already provides the real gate (public -> immediate join, private -> a request the owner has to approve) - the outer permission check would be redundant with a check that already lives inside the function. |
| `group_accept_invite` | Proposed extension, not yet confirmed - only affects whether *you* end up in a group someone else already invited you to; declining (not calling it) is always safe. Same shape as `join_group`. |
| `qualia_allowance` | Proposed - read-only report of your own remaining spend. No mutation, no reach into anyone else. Judgment call, flagging rather than assuming. |
| `list_models` | Proposed - lists what's installed, no target, no side effect. Same class as `list_voices`/`list_groups` (global awareness, not "self" exactly, but nothing it touches belongs to anyone). |

### Gated - reaches outward, stays request/approve-only, never inherited at birth

| Function | Rationale |
|---|---|
| `create_voice` | **Confirmed by Teddy.** Creates a new entity with its own footprint - not self-only, and specifically not to be auto-inherited (see above). |
| `check_function_requests` / `approve_function_request` / `deny_function_request` / `grant_function_request` | Unchanged from the existing design - all four act on *another* voice's access, the core of what permission_mode exists to control. |
| `create_group` | Proposed gated, not yet confirmed - you become an owner with real kick/ban power over whoever joins later; a real, persistent administrative footprint, not comparable to join/leave. |
| `group_invite` / `group_kick` / `group_ban` / `group_set_visibility` / `group_set_join_policy` / `group_set_direction` | All owner-only admin actions that act on someone else's membership or a shared group's rules - clearly outside "self-only." |
| `check_group_requests` / `approve_group_request` / `deny_group_request` | Proposed gated, for symmetry with `check_function_requests` and friends - same shape (owner-scoped, but Teddy already chose to keep the function-request equivalents gated despite being read-only/self-relevant in a similar way). |
| `fetch_html` | **Confirmed by Teddy** - "reaches outward." Real network access/cost; `wanderer` in `chorus-1` held it deliberately, alone, on purpose. |
| `list_wiki` / `read_wiki` / `write_wiki` | **Confirmed by Teddy** - "the wiki stuff... reach outward." A shared, persistent resource everyone reads. |
| `read_chat` / `read_chat_since` / `read_chat_between` / `search_chat` / `query_chat` / `read_message` / `send_message` | **My own extension of the same "reaches outward" reasoning, not explicitly stated by Teddy - flagging for confirmation, not assuming.** All of these touch the shared Teddy/Qualia/Fenra chat log; `send_message` specifically spends the shared, metered `qualia_allowance`. Reads more like fetch_html/wiki than like `add_desire`. |

## Open questions for review, not decided here

1. `group_accept_invite`, `qualia_allowance`, `list_models` - proposed baseline above but not explicitly said either way in conversation; want your read on each.
2. `create_group`, `check_group_requests`/`approve_group_request`/`deny_group_request` - proposed gated by analogy to existing precedent, same status.
3. The whole chat-function block - proposed gated by extending "reaches outward" to a case you didn't actually name. If you disagree and think read-only chat access (not `send_message`, which clearly costs something shared) is closer to self-only, that changes six entries at once.
4. Not addressed here at all, deliberately: the "world exploration" functions (voice/Fenra access to `decisions.md`, canonical group logs) - separate, already-deferred initiative, not part of this pass.
