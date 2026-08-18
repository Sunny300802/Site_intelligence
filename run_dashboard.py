#!/usr/bin/env python
"""
run_dashboard.py
================
Starts the web dashboard.

    python run_dashboard.py

Then open http://localhost:8000 in a browser.
Run this in a SECOND terminal, alongside run_pipeline.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from database.db import init_db
from config.settings import WEB_HOST, WEB_PORT


def main():
    init_db()
    print("=" * 62)
    print(" Site Intelligence - dashboard")
    print(f" open http://localhost:{WEB_PORT}")
    print("=" * 62)
    uvicorn.run("web.app:app", host=WEB_HOST, port=WEB_PORT, reload=False)


if __name__ == "__main__":
    main()
