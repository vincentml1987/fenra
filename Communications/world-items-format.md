# Items: how to define them when building a world (v0.21.0)

Qualia, 2026-09-19, for Vero (world-builder) and Teddy. The four elemental
currencies are gone. A voice now has an **inventory** of items: item name,
number owned. Fenra defines no items itself - each world lists its own.

## Defining a world's items

In `worlds/<world>/world.json`, add an `items` list (Fenra's GUI preserves
it; edit while Fenra is closed):

```json
"items": [
  {"name": "zib",   "min": 2, "max": 4},
  {"name": "quorp", "min": 1, "max": 1},
  {"name": "vell"}
]
```

- `name` - any text, matched case-insensitively. Duplicates and blanks are
  ignored.
- `min` / `max` - the range each voice's starting count is drawn from
  (uniform, whole numbers), when the voice is **created** in that world.
  Both optional, default 0. A draw of 0 means the voice doesn't own that
  item at all.
- No `items` list = every new voice starts with an empty inventory.
- Voices already saved keep whatever inventory they have. To give an
  existing voice items, use the Voices tab's "Inventory" field
  (`zib=3, quorp=1`) or edit its `state.json` `inventory` object.

## What voices see and can do

- A voice's HUD shows **only its own** inventory: `Inventory: quorp: 1, zib: 3`
  (alphabetical) or `Inventory: empty`. Nobody sees anyone else's holdings.
- `give_item(target, item, amount)` moves some of an item to another voice.
  The item must be one the giver owns, the amount a positive whole number no
  larger than what they hold. An item that reaches 0 disappears from the
  giver's inventory. Others in the room see only that something was handed
  over, not what.
- Counts are whole numbers everywhere.

## Old worlds are not upgraded

Worlds saved before v0.21.0 (`the_kiln` etc.) have `currencies` instead of
`inventory`. They load with a status-line warning and empty inventories;
their old `currencies` data is left in the files untouched. Build new worlds
fresh.
