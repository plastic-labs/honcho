#!/usr/bin/env python
"""
Utility script to generate a JWT secret for use in the .env file.
This uses the same logic as the automatically generated version in security.py.
"""

import argparse
import secrets


def generate_jwt_secret():
    """Generate a random JWT secret using the secrets module."""
    return secrets.token_hex(32)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a JWT secret for authentication."
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the secret without instructions",
    )
    args = parser.parse_args()

    secret = generate_jwt_secret()

    if args.print_only:
        print(secret)
    else:
        print(f"Generated JWT secret: {secret}")
        print("\nAdd this to your .env file as:")
        print(f"AUTH_JWT_SECRET={secret}")
        print(f"or as WEBHOOK_SECRET={secret}")


if __name__ == "__main__":
    main()
