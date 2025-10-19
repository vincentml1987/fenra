# DEPRECATED: Discord is now embedded in fenra_ui.py (in-memory only, no file queue).
#!/usr/bin/env python3
import os
import json
from datetime import datetime
import discord

from config_loader import load_globals
from pdv_utils import apply_and_persist_pdv_adjustments

# ─── configuration ────────────────────────────────────────────────────────────
# your bot token (you said you set fenra_token as a system variable)
DISCORD_TOKEN = os.getenv("fenra_token")
# the numeric channel ID for #chat-with-fenra (enable Dev Mode → Copy Channel ID)
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))
# where Fenra expects queued messages
QUEUE_PATH = os.path.join("chatlogs", "queued_messages.json")

if not DISCORD_TOKEN or CHANNEL_ID == 0:
    raise RuntimeError("set fenra_token and DISCORD_CHANNEL_ID env vars before running")

# ─── helpers ─────────────────────────────────────────────────────────────────
def load_queue():
    try:
        with open(QUEUE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_queue(q):
    os.makedirs(os.path.dirname(QUEUE_PATH), exist_ok=True)
    with open(QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(q, f, ensure_ascii=False, indent=2)

# ─── Discord client ─────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True  # required to read messages

class DiscordToFenra(discord.Client):
    async def on_ready(self):
        print(f"[Discord→Fenra] Logged in as {self.user} (listening on {CHANNEL_ID})")

    async def on_message(self, msg):
        # ignore bots (including itself) and other channels
        if msg.author.bot or msg.channel.id != CHANNEL_ID:
            return

        # Use display_name (nickname in that server, or username fallback)
        author = msg.author.display_name

        # Build a single‐field JSON entry where "message" includes the author
        entry = {
            "timestamp": msg.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "message": (
                f"The following message was sent by Discord user {author}: "
                f"{msg.content}"
            )
        }

        queue = load_queue()
        queue.append(entry)
        save_queue(queue)

        # Legacy incoming_message_*pdvms removed — only Discord-controlled PDV adjustments remain.
        # Discord-controlled PDV based on message length
        try:
            g = load_globals() or {}
            pdv_name = (g.get("discord_pdv_name") or "").strip()
            scale_val = g.get("discord_pdv_scale", 0.0)
            scale = float(scale_val or 0.0)
            if scale < 0:
                scale = 0.0
            elif scale > 1:
                scale = 1.0
            if pdv_name and scale > 0:
                delta = len(msg.content or "") * scale
                if delta != 0:
                    apply_and_persist_pdv_adjustments([{"name": pdv_name, "delta": delta}])
                    print(f"[PDVM] Discord adjusted {pdv_name} by +{delta:.3f}")
        except Exception as e:
            print(f"[Discord→Fenra] Discord PDV update failed: {e}")

        print(f"[Discord→Fenra] Queued message at {entry['timestamp']}")


# ─── run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = DiscordToFenra(intents=intents)
    client.run(DISCORD_TOKEN)
