#!/usr/bin/env python3
import os
import json
from datetime import datetime
import discord

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

        # build the Fenra‑style queue entry
        entry = {
            "sender": msg.author.name,
            "timestamp": msg.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "message": msg.content,
            # optional: you could add "epoch": msg.created_at.timestamp(),
            # and/or a "groups": ["general"] key if you like.
        }

        queue = load_queue()
        queue.append(entry)
        save_queue(queue)
        print(f"[Discord→Fenra] Queued message from {entry['sender']} at {entry['timestamp']}")

# ─── run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = DiscordToFenra(intents=intents)
    client.run(DISCORD_TOKEN)
