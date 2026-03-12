#!/usr/bin/env python3
"""Load Gmail messages into Honcho.

Uses the Gmail API directly (with OAuth) to fetch emails and the Honcho Python SDK to store them.
Each Gmail thread becomes a Honcho session, each sender becomes a peer.

Prerequisites:
1. Create a Google Cloud project and enable the Gmail API
2. Create OAuth 2.0 credentials (Desktop app type)
3. Download the credentials JSON and save as 'credentials.json' in this directory
4. Install dependencies:
   pip install google-api-python-client google-auth-oauthlib honcho-ai

On first run, a browser window will open for OAuth consent. After authorizing,
a 'token.json' file will be created to store your credentials for future runs.
"""

import argparse
import base64
import os
import re
import time
from datetime import datetime, timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_gmail_service(credentials_file: str = "credentials.json", token_file: str = "token.json"):
    """Authenticate and return a Gmail API service instance."""
    creds = None

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_file):
                raise FileNotFoundError(
                    f"Credentials file '{credentials_file}' not found.\n"
                    "Download OAuth credentials from Google Cloud Console:\n"
                    "1. Go to console.cloud.google.com\n"
                    "2. Create/select a project and enable Gmail API\n"
                    "3. Create OAuth 2.0 credentials (Desktop app)\n"
                    "4. Download JSON and save as 'credentials.json'"
                )
            print("Opening browser for OAuth consent...")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_file, "w") as token:
            token.write(creds.to_json())
        print(f"Credentials saved to {token_file}")

    return build("gmail", "v1", credentials=creds)


def list_threads(service, query: str = None, label_ids: list = None, max_results: int = 10) -> list[dict]:
    """List Gmail threads with pagination support."""
    all_threads = []
    page_token = None

    while len(all_threads) < max_results:
        try:
            params = {
                "userId": "me",
                "maxResults": min(100, max_results - len(all_threads)),
            }
            if query:
                params["q"] = query
            if label_ids:
                params["labelIds"] = label_ids
            if page_token:
                params["pageToken"] = page_token

            response = service.users().threads().list(**params).execute()
            threads = response.get("threads", [])
            all_threads.extend(threads)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        except HttpError as e:
            print(f"Error listing threads: {e}")
            break

    return all_threads[:max_results]


def get_thread(service, thread_id: str) -> dict:
    """Fetch a complete Gmail thread with all messages."""
    try:
        return service.users().threads().get(
            userId="me",
            id=thread_id,
            format="full"
        ).execute()
    except HttpError as e:
        print(f"Error fetching thread {thread_id}: {e}")
        return {}


def extract_email(from_header: str) -> str:
    """Extract bare email from 'Name <email>' format."""
    match = re.search(r"<([^>]+)>", from_header)
    return match.group(1).lower() if match else from_header.lower().strip()


def extract_name(from_header: str) -> str:
    """Extract display name from 'Name <email>' format."""
    match = re.match(r'^"?([^"<]+)"?\s*<', from_header)
    return match.group(1).strip() if match else from_header.strip()


def decode_body(payload: dict) -> str:
    """Recursively extract plain text from a Gmail message payload."""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    parts = payload.get("parts", [])
    for part in parts:
        text = decode_body(part)
        if text:
            return text
    return ""


def strip_quoted_replies(text: str) -> str:
    """Strip quoted reply text from an email body, keeping only the new content."""
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^On .+wrote:\s*$", stripped):
            break
        if stripped.startswith("---------- Forwarded message"):
            break
        if stripped.startswith(">"):
            break
        if re.match(r"^[-_]{10,}$", stripped):
            break
        clean_lines.append(line)
    return "\n".join(clean_lines).rstrip()


def parse_address_list(header: str) -> list[str]:
    """Parse a comma-separated email header into individual addresses."""
    if not header.strip():
        return []
    parts = re.split(r",(?![^<]*>)", header)
    return [p.strip() for p in parts if p.strip()]


def peer_id_from_email(email: str) -> str:
    """Convert email to a valid Honcho peer ID."""
    return email.replace("@", "-").replace(".", "-")


def fetch_thread_messages(service, thread_id: str) -> list[dict]:
    """Fetch all messages in a Gmail thread with full content."""
    data = get_thread(service, thread_id)
    messages = []

    for msg in data.get("messages", []):
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        body = strip_quoted_replies(decode_body(msg.get("payload", {})))
        ts = int(msg.get("internalDate", "0")) / 1000

        messages.append({
            "id": msg["id"],
            "thread_id": msg["threadId"],
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "cc": headers.get("Cc", ""),
            "bcc": headers.get("Bcc", ""),
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc),
            "body": body.strip(),
            "labels": msg.get("labelIds", []),
            "snippet": msg.get("snippet", ""),
        })

    return messages


def main():
    parser = argparse.ArgumentParser(description="Load Gmail messages into Honcho")
    parser.add_argument("--workspace", "-w", default="gmail", help="Honcho workspace ID (default: gmail)")
    parser.add_argument("--query", "-q", default=None, help="Gmail search query (e.g. 'from:alice@example.com')")
    parser.add_argument("--label", "-l", default=None, help="Gmail label to filter by (e.g. INBOX)")
    parser.add_argument("--max-threads", "-n", type=int, default=10, help="Max threads to fetch (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be loaded without writing to Honcho")
    parser.add_argument("--credentials", "-c", default="credentials.json", help="Path to OAuth credentials JSON")
    parser.add_argument("--token", "-t", default="token.json", help="Path to store/load access token")
    args = parser.parse_args()

    # Authenticate
    print("Authenticating with Gmail API...")
    service = get_gmail_service(args.credentials, args.token)
    print("  Authenticated successfully!")

    label_ids = [args.label] if args.label else None

    # List threads
    print(f"\nFetching up to {args.max_threads} threads from Gmail...")
    threads = list_threads(service, query=args.query, label_ids=label_ids, max_results=args.max_threads)
    print(f"  Found {len(threads)} threads")

    if not threads:
        print("No threads found. Try adjusting --query or --label.")
        return

    # Fetch full messages for each thread
    all_thread_messages = {}
    seen_peers = {}

    def register_peer(addr: str):
        email = extract_email(addr)
        if email and email not in seen_peers:
            name = extract_name(addr)
            if name.lower().strip() == email or "@" in name:
                name = email.split("@")[0].replace(".", " ").title()
            seen_peers[email] = {
                "name": name,
                "peer_id": peer_id_from_email(email),
                "email": email,
            }

    for i, t in enumerate(threads):
        tid = t["id"]
        print(f"  Fetching thread {i+1}/{len(threads)}: {tid}")
        msgs = fetch_thread_messages(service, tid)
        all_thread_messages[tid] = msgs
        for m in msgs:
            register_peer(m["from"])
            for addr in parse_address_list(m["to"]):
                register_peer(addr)
            for addr in parse_address_list(m["cc"]):
                register_peer(addr)
            for addr in parse_address_list(m["bcc"]):
                register_peer(addr)

    # Summary
    total_msgs = sum(len(v) for v in all_thread_messages.values())
    print(f"\nSummary:")
    print(f"  Threads: {len(all_thread_messages)}")
    print(f"  Messages: {total_msgs}")
    print(f"  Unique participants: {len(seen_peers)}")
    for email, info in seen_peers.items():
        print(f"    {info['peer_id']} ({info['name']} <{email}>)")

    if args.dry_run:
        print("\n[DRY RUN] Would create the above in Honcho. Showing first message per thread:")
        for tid, msgs in all_thread_messages.items():
            m = msgs[0]
            body_preview = m["body"][:120].replace("\n", " ") if m["body"] else m["snippet"][:120]
            print(f"  Thread {tid}: {m['subject']}")
            print(f"    {m['from']} @ {m['date']}")
            print(f"    {body_preview}...")
        return

    # Load into Honcho
    from honcho import Honcho

    print(f"\nLoading into Honcho workspace '{args.workspace}'...")
    honcho = Honcho(workspace_id=args.workspace)

    # Create peers
    peers = {}
    for i, (email, info) in enumerate(seen_peers.items()):
        if i > 0 and i % 4 == 0:
            time.sleep(1)
        peers[email] = honcho.peer(info["peer_id"], metadata={
            "email": email,
            "name": info["name"],
            "source": "gmail",
        })
        print(f"  Peer: {info['peer_id']}")

    # Create sessions and messages per thread
    for tid, msgs in all_thread_messages.items():
        subject = msgs[0]["subject"] if msgs else "No subject"
        session_id = f"gmail-thread-{tid}"

        thread_peer_emails = set()
        for m in msgs:
            thread_peer_emails.add(extract_email(m["from"]))
            for addr in parse_address_list(m["to"]):
                thread_peer_emails.add(extract_email(addr))
            for addr in parse_address_list(m["cc"]):
                thread_peer_emails.add(extract_email(addr))
            for addr in parse_address_list(m["bcc"]):
                thread_peer_emails.add(extract_email(addr))
        thread_peers = [peers[e] for e in thread_peer_emails if e in peers]

        session = honcho.session(session_id, metadata={
            "gmail_thread_id": tid,
            "subject": subject,
            "source": "gmail",
            "message_count": len(msgs),
        })
        session.add_peers(thread_peers)

        honcho_msgs = []
        for m in msgs:
            email = extract_email(m["from"])
            peer = peers.get(email)
            if not peer:
                continue
            content = m["body"] if m["body"] else m["snippet"]
            if not content:
                continue
            honcho_msgs.append(peer.message(
                content,
                metadata={
                    "gmail_id": m["id"],
                    "subject": m["subject"],
                    "from": m["from"],
                    "to": m["to"],
                    "labels": m["labels"],
                },
                created_at=m["timestamp"],
            ))

        if honcho_msgs:
            session.add_messages(honcho_msgs)
            print(f"  Session {session_id}: {len(honcho_msgs)} messages — {subject[:60]}")

    print(f"\nDone! Loaded {total_msgs} messages into workspace '{args.workspace}'.")


if __name__ == "__main__":
    main()
