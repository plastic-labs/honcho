# Discord connector

Reads a Discord server's history as Honcho ingest records, and writes them to
JSONL files or sends them to the ingest API. Run from the Honcho CLI with
`--follow`, it then follows the server live: new messages, edits and deletes,
with its position saved in Honcho. It has no install flow yet: it runs with
your own bot token.

> **Experimental.** Ingest into auto workspaces is experimental, and so is
> this connector: its records, its options and the ingest API it sends to may
> change or stop working without notice. The CLI commands that import and review
> its output run only with `HONCHO_EXPERIMENTAL=1` set.

## What it needs

A bot token, in a file, for a bot that has been added to the server with:

- **View Channels** and **Read Message History** in the channels to export;
- the **Message Content** intent, or message text comes back empty;
- optionally the **Server Members** intent, for nicknames. Without it, names
  fall back to each account's global display name.

An export only uses Discord's HTTP API and never opens a gateway connection, so
it can run beside a bot that is already connected with the same token.
Following opens one (as a second session of the same bot, if it's running).

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

# Or, installed next to the Honcho CLI, read and import in one step
# with the CLI's own workspace and key
export HONCHO_EXPERIMENTAL=1
honcho ingest from discord --token-file ~/.discord.token --guild GUILD_ID -w discord
```

The package registers itself as the `discord` source for `honcho ingest from`
(`pip install honcho-cli ./connectors/discord`). There the CLI sends the records
and holds back bot records no person interacted with (`--bots`).

`--channels-file` takes one channel ID per line (`#` starts a comment). A
channel's threads, active and archived, come with it. Private archived threads
aren't read.

## Following a server

```bash
export HONCHO_EXPERIMENTAL=1
honcho ingest from discord --follow --token-file ~/.discord.token --guild GUILD_ID -w discord
```

It reads the history from the position it saved (all of it the first time),
then follows live events until you stop it with Ctrl-C or SIGTERM, when it sends
what's pending and saves its position. Run it again, on any machine, and it
carries on from there. Records are sent with `--derive auto`: stored for later
during the dry run, derived as they arrive after it.

- **New messages** are never missed: each channel's and thread's position moves
  only past messages Honcho accepted (or that were skipped on purpose). After a
  restart, a dropped connection or falling behind, it reads the history again
  from there. A message created and deleted while it was down isn't seen.
- **Edits and deletes** are taken while it's connected. They change stored
  records only during the dry run; after it, Honcho refuses them
  (`edited_after_derivation`). One made while it was down isn't seen.
- **New threads and channels** under the selected channels are followed as they
  appear.
- **Bot messages** wait up to `--bot-window` minutes (60 by default) for a person
  to reply to or mention the bot; unanswered, they're dropped. While it reads
  history, the wait is measured in the history's own time.
- **One runner per server.** A second runner, or a reset that forgot the
  position, is noticed at the next save: it stops with `SYNC_CONFLICT`. It isn't
  a lock: two first runs both start.

`honcho ingest status` shows the server's last sync, newest record and last
error.

## What a record holds

| Ingest field | From Discord |
|---|---|
| `source`, `source_instance` | `discord`, the server ID |
| `source_id`, `created_at`, `revised_at` | the message ID, its timestamp, its last edit |
| `container` | the channel or thread. `parent_id` is a thread's channel, or a channel's category |
| `author` | the account ID, the name the server shows (nickname, global name, username), and `bot: true` for bots and webhooks |
| `participants` | the accounts it mentions, and the author of the message it replies to, bots marked the same way |
| `content` | the text as written; mentions stay `<@id>`. A message without text keeps a short note of its embeds, attachments, stickers or poll |
| `metadata` | the message type, the replied-to message ID, attachment names |

System notices (joins, pins, thread created) aren't records. Bot and webhook
messages are; Honcho stores them, and a bot's peer is created with nothing
derived about it. A peer that already existed before its first flagged record
keeps its configuration.

Files written by `--out`: `records.jsonl` (ingest records, in send order),
`accounts.jsonl` (each account's ID, username, global name, nickname and bot
flag), `containers.jsonl`, and `export.json` (counts and category names).

## Limits

- Discord's history API returns each message's current text only, and never a
  deleted message, so an export has no edit history or deletions. Following
  sees edits and deletes only while it's connected.
- Deleting a whole channel or thread isn't followed: Discord sends no message
  IDs for it.
- A link preview Discord adds after a message is sent isn't followed.
- Names are as of the export, not as of each message.
