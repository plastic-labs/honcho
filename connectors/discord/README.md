# Discord connector

Reads a Discord server's history as Honcho ingest records, and writes them to
JSONL files or sends them to the ingest API. This is the backfill half of a
Discord connector: it reads history once, oldest first. It has no live events,
install flow or stored read position yet.

## What it needs

A bot token, in a file, for a bot that has been added to the server with:

- **View Channels** and **Read Message History** in the channels to export;
- the **Message Content** intent, or message text comes back empty;
- optionally the **Server Members** intent, for nicknames. Without it, names
  fall back to each account's global display name.

It only uses Discord's HTTP API and never opens a gateway connection, so it can
run beside a bot that is already connected with the same token.

## Use

```bash
# The channels an export would read: id, category, name, type, thread count
honcho-discord channels --token-file ~/.discord.token --guild GUILD_ID

# Export to files (refuses a directory inside a git checkout)
honcho-discord export --token-file ~/.discord.token --guild GUILD_ID \
    [--channel ID ... | --channels-file FILE] --out ~/discord-export

# Or send to Honcho's ingest API, deriving later
HONCHO_API_KEY=... honcho-discord export --token-file ~/.discord.token \
    --guild GUILD_ID --ingest-url http://127.0.0.1:8000 --workspace discord \
    --api-key-env HONCHO_API_KEY
```

`--channels-file` takes one channel ID per line (`#` starts a comment). A
channel's threads, active and archived, come with it. Private archived threads
aren't read.

## What a record holds

| Ingest field | From Discord |
|---|---|
| `source`, `source_instance` | `discord`, the server ID |
| `source_id`, `created_at`, `revised_at` | the message ID, its timestamp, its last edit |
| `container` | the channel or thread. `parent_id` is a thread's channel, or a channel's category |
| `author` | the account ID and the name the server shows (nickname, global name, username) |
| `participants` | the accounts it mentions, and the author of the message it replies to |
| `content` | the text as written; mentions stay `<@id>`. A message without text keeps a short note of its embeds, attachments, stickers or poll |
| `metadata` | the message type, the replied-to message ID, attachment names, and `bot` for bots and webhooks |

System notices (joins, pins, thread created) aren't records. Bot and webhook
messages are, marked `bot`.

Files written by `--out`: `records.jsonl` (ingest records, in send order),
`accounts.jsonl` (each account's ID, username, global name, nickname and bot
flag), `containers.jsonl`, and `export.json` (counts and category names).

## Limits

- Discord's history API returns each message's current text only, and never a
  deleted message, so an export has no edit history or deletions.
- Names are as of the export, not as of each message.
