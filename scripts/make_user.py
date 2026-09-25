"""
Create or update a sign-in account.

Passwords are never stored - only a PBKDF2-HMAC-SHA256 hash with a per-user salt, which is
what goes into configs/users.yaml. The password itself is typed here and forgotten.

Usage:
    python -m scripts.make_user                       # prompts for everything
    python -m scripts.make_user --username ana --role supervisor
    python -m scripts.make_user --username ana --role supervisor --password "..."   # avoid
        (the last form puts the password in your shell history - use it only in scripts)

Roles:
    viewer      see the site; cannot decide anything
    supervisor  approve or reject what the copilot proposes
    admin       everything, plus configuration and maintenance
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from server.auth import ADMIN, ROLE_ORDER, SUPERVISOR, VIEWER, hash_password

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = ROOT / "configs" / "users.yaml"
MIN_PASSWORD = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--username")
    parser.add_argument("--role", choices=sorted(ROLE_ORDER), default=SUPERVISOR)
    parser.add_argument("--display-name", default="")
    parser.add_argument("--password", help="skips the prompt; ends up in shell history")
    parser.add_argument("--list", action="store_true", help="show accounts (never hashes)")
    return parser


def load(path: Path) -> dict:
    if not path.exists():
        return {"users": []}
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"users": []}


def save(path: Path, data: dict) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Sign-in accounts. Password HASHES only - never a password.\n"
              "# Created with: python -m scripts.make_user\n"
              "# This file is gitignored: accounts are per-machine, not source.\n")
    path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def upsert(data: dict, username: str, role: str, password: str, display_name: str) -> str:
    entry = {"username": username, "role": role, "display_name": display_name or username}
    entry.update(hash_password(password))
    users = data.setdefault("users", [])
    for index, existing in enumerate(users):
        if str(existing.get("username", "")).lower() == username:
            users[index] = entry
            return "updated"
    users.append(entry)
    return "created"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data = load(args.file)

    if args.list:
        users = data.get("users", [])
        if not users:
            print(f"No accounts in {args.file}")
            return 0
        print(f"{'username':<20}{'role':<14}display name")
        for user in users:
            print(f"{user.get('username',''):<20}{user.get('role',''):<14}{user.get('display_name','')}")
        return 0

    username = (args.username or input("username: ")).strip().lower()
    if not username:
        print("A username is required.")
        return 1

    password = args.password or getpass.getpass("password: ")
    if not args.password:
        if password != getpass.getpass("repeat password: "):
            print("The two passwords do not match.")
            return 1
    if len(password) < MIN_PASSWORD:
        print(f"Use at least {MIN_PASSWORD} characters.")
        return 1

    what = upsert(data, username, args.role, password, args.display_name)
    save(args.file, data)
    print(f"{what} {username} ({args.role}) in {args.file}")
    print("The password was hashed and discarded; only the hash is stored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
