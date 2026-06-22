#!/usr/bin/env uv run python
"""
Utility script to generate scoped JWTs for Honcho.

Examples:
    # Admin JWT (no expiry)
    uv run python scripts/generate_jwt.py --admin

    # Admin JWT expiring in 24 hours
    uv run python scripts/generate_jwt.py --admin --expires 24h

    # Workspace-scoped JWT expiring in 30 days
    uv run python scripts/generate_jwt.py --workspace my-workspace --expires 30d

    # Peer-scoped JWT expiring in 1 year
    uv run python scripts/generate_jwt.py --workspace my-workspace --peer my-peer --expires 1y

    # Session-scoped JWT
    uv run python scripts/generate_jwt.py --workspace my-workspace --session my-session --expires 8h
"""

import argparse
import datetime
import os
import re
import sys

# Allow running from repo root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.security import JWTParams, create_jwt
from src.utils.formatting import format_datetime_utc

DURATION_UNITS = {
    "s": datetime.timedelta(seconds=1),
    "m": datetime.timedelta(minutes=1),
    "h": datetime.timedelta(hours=1),
    "d": datetime.timedelta(days=1),
    "w": datetime.timedelta(weeks=1),
    "y": datetime.timedelta(days=365),
}


def parse_duration(value: str) -> datetime.timedelta:
    """Parse a duration string like '5h', '1d', '2w', '1y' into a timedelta."""
    match = re.fullmatch(r"(\d+)([smhdwy])", value.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid duration '{value}'. Use format like: 30s, 5m, 2h, 7d, 2w, 1y"
        )
    amount, unit = int(match.group(1)), match.group(2)
    return DURATION_UNITS[unit] * amount


def main():
    parser = argparse.ArgumentParser(
        description="Generate a scoped JWT for Honcho authentication.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--admin",
        action="store_true",
        help="Generate an admin JWT (full access)",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        metavar="NAME",
        help="Scope the JWT to a workspace",
    )
    parser.add_argument(
        "--peer",
        "-p",
        metavar="NAME",
        help="Scope the JWT to a peer (requires --workspace)",
    )
    parser.add_argument(
        "--session",
        "-s",
        metavar="NAME",
        help="Scope the JWT to a session (requires --workspace)",
    )
    parser.add_argument(
        "--expires",
        "-e",
        metavar="DURATION",
        type=parse_duration,
        help="Token expiry duration. Units: s=seconds, m=minutes, h=hours, d=days, w=weeks, y=years. E.g. 5h, 30d, 1y",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the token, no labels",
    )
    args = parser.parse_args()

    if not args.admin and not any([args.workspace, args.peer, args.session]):
        parser.error(
            "Specify --admin or at least one of --workspace, --peer, --session"
        )

    if args.admin and any([args.workspace, args.peer, args.session]):
        parser.error(
            "--admin cannot be combined with --workspace, --peer, or --session"
        )

    if (args.peer or args.session) and not args.workspace:
        parser.error("--peer and --session require --workspace")

    exp_str: str | None = None
    if args.expires:
        expiry = datetime.datetime.now(datetime.timezone.utc) + args.expires
        exp_str = format_datetime_utc(expiry)

    params = JWTParams(
        ad=True if args.admin else None,
        w=args.workspace,
        p=args.peer,
        s=args.session,
        exp=exp_str,
    )

    token = create_jwt(params)

    if args.print_only:
        print(token)
    else:
        scope_parts: list[str] = []
        if args.admin:
            scope_parts.append("admin")
        if args.workspace:
            scope_parts.append(f"workspace={args.workspace}")
        if args.peer:
            scope_parts.append(f"peer={args.peer}")
        if args.session:
            scope_parts.append(f"session={args.session}")

        print(f"Scope:   {', '.join(scope_parts)}")
        if exp_str:
            print(f"Expires: {exp_str}")
        else:
            print("Expires: never")
        print(f"Token:   {token}")


if __name__ == "__main__":
    main()
