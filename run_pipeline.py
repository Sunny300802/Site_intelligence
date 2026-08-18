#!/usr/bin/env python
"""
run_pipeline.py
===============
Starts the vision engine: reads the cameras, detects and tracks people,
recognises faces, writes to the database and streams annotated video.

    python run_pipeline.py

Leave this running. Press Ctrl+C to stop it cleanly.

"""
import warnings

warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pipeline import Pipeline


def main():
    print("=" * 62)
    print(" Site Intelligence - vision pipeline")
    print("=" * 62)
    pipeline = Pipeline()
    pipeline.setup()
    pipeline.run()


if __name__ == "__main__":
    main()
