"""
tools/reset_enrollment.py
=========================
Throw away everything the system has LEARNED and start enrollment again.

    python tools/reset_enrollment.py              # say what would go
    python tools/reset_enrollment.py --apply      # actually do it

What goes
---------
    data/faces/*/cctv_*.jpg     the photographs the system took of
                                itself, and every vector derived from
                                them
    data/face_encodings.pkl     the enrollment index built from the
                                photos
    employees                   the enrolled person list in the database
    learned_faces               face vectors confirmed on the dashboard
                                or captured automatically
    learned_appearance          clothing signatures - no longer used at
                                all, see core/body.py
    review_items + data/review  the "is this X?" queue and its crops

What STAYS
----------
Your own photographs - anything in data/faces that this system did not
write. They are the only thing here that cannot be regenerated, so they
are never touched, and re-enrolling is one command afterwards.

Attendance (visits, presence) also stays unless you pass --wipe-history.
That is a record of who was in the building, not something the system
learned about them.

Everything deleted is copied to data/backup_<timestamp>/ first. A reset
that cannot be undone is a reset nobody dares run.
"""
import os
import sys
import glob
import shutil
import sqlite3
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DATA_DIR, FACE_DIR, DB_PATH, ENCODINGS_FILE, REVIEW_DIR

# Written by core/face_library.py. The prefix is what separates a
# photograph the system took from one a person took.
CAPTURE_PREFIX = "cctv_"

# Tables that hold LEARNED data. Attendance is deliberately not here.
LEARNED_TABLES = ["employees", "learned_faces", "learned_appearance",
                  "review_items"]
HISTORY_TABLES = ["visits", "presence"]


def captures():
    return sorted(glob.glob(os.path.join(FACE_DIR, "*", CAPTURE_PREFIX + "*")))


def review_files():
    out = []
    for root, _dirs, names in os.walk(REVIEW_DIR):
        out.extend(os.path.join(root, n) for n in names)
    return sorted(out)


def counts(db_path, tables):
    if not os.path.exists(db_path):
        return {}
    out = {}
    con = sqlite3.connect(db_path)
    try:
        for table in tables:
            try:
                out[table] = con.execute(
                    f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error:
                out[table] = 0
    finally:
        con.close()
    return out


def backup(paths, tag):
    """Copy everything about to be deleted into one folder."""
    root = os.path.join(DATA_DIR, f"backup_{tag}")
    os.makedirs(root, exist_ok=True)
    kept = 0
    for path in paths:
        if not os.path.exists(path):
            continue
        relative = os.path.relpath(path, DATA_DIR)
        target = os.path.join(root, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(path, target)
        kept += 1
    return root, kept


def main():
    parser = argparse.ArgumentParser(
        description="Delete enrolled and self-trained data, keeping your "
                    "own photographs")
    parser.add_argument("--apply", action="store_true",
                        help="actually delete (default is a dry run)")
    parser.add_argument("--wipe-history", action="store_true",
                        help="also delete visits and presence rows")
    parser.add_argument("--keep-reviews", action="store_true",
                        help="leave the review queue and its crops alone")
    args = parser.parse_args()

    tables = list(LEARNED_TABLES)
    if args.keep_reviews:
        tables.remove("review_items")
    if args.wipe_history:
        tables += HISTORY_TABLES

    photos = captures()
    crops = [] if args.keep_reviews else review_files()
    rows = counts(DB_PATH, tables)
    own = len([p for p in glob.glob(os.path.join(FACE_DIR, "*", "*"))
               if os.path.isfile(p) and
               not os.path.basename(p).startswith(CAPTURE_PREFIX)])

    print("=" * 64)
    print(" RESET - enrolled and self-trained data")
    print("=" * 64)
    print(f"  camera-captured photos   {len(photos)}")
    print(f"  review crops             {len(crops)}")
    print(f"  encodings file           "
          f"{'yes' if os.path.exists(ENCODINGS_FILE) else 'missing'}")
    for table, n in rows.items():
        print(f"  {table:<24} {n} row(s)")
    print(f"\n  YOUR OWN PHOTOGRAPHS      {own} - kept, untouched")
    if not args.wipe_history:
        print("  attendance history        kept "
              "(pass --wipe-history to clear)")

    if not args.apply:
        print("\nDry run. Nothing was deleted. Re-run with --apply.")
        return 0

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    root, kept = backup([DB_PATH, ENCODINGS_FILE] + photos + crops, tag)
    print(f"\n[RESET] backed up {kept} file(s) -> {root}")

    for path in photos + crops:
        try:
            os.remove(path)
        except OSError as exc:
            print(f"[RESET] could not delete {path}: {exc}")
    if os.path.exists(ENCODINGS_FILE):
        os.remove(ENCODINGS_FILE)
    print(f"[RESET] deleted {len(photos)} captured photo(s), "
          f"{len(crops)} review crop(s), the encodings file")

    if os.path.exists(DB_PATH):
        con = sqlite3.connect(DB_PATH)
        try:
            for table in tables:
                try:
                    con.execute(f"DELETE FROM {table}")
                except sqlite3.Error as exc:
                    print(f"[RESET] {table}: {exc}")
            con.commit()
            con.execute("VACUUM")
        finally:
            con.close()
        print(f"[RESET] cleared {', '.join(tables)}")

    print("\n[RESET] done. Now enroll again:")
    print("    python tools/enroll_faces.py")
    print("    python run_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
