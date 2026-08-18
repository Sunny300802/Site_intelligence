"""
tools/init_database.py
======================
Create the database and the dashboard login.

    python tools/init_database.py --username admin --password "YourPass#1"

Safe to run again later: it only creates what is missing and never
deletes recorded data.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import init_db
from database import repository as repo
from web.auth import hash_password


def main(username, password, role):
    init_db()
    print("[INIT] database and tables ready")
    if repo.get_user(username):
        print(f"[INIT] user '{username}' already exists - nothing else to do")
        return 0
    repo.create_user(username, hash_password(password), role)
    print(f"[INIT] created {role} user '{username}'")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--role", default="admin")
    a = ap.parse_args()
    if a.password == "admin":
        print("[INIT] WARNING: using the default password 'admin'. "
              "Re-run with --password to set a real one.")
    raise SystemExit(main(a.username, a.password, a.role))
