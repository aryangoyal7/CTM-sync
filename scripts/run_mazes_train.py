#!/usr/bin/env python3
"""Run tasks.mazes.train from any working directory.

Usage:
  /path/to/venv/bin/python /path/to/repo/scripts/run_mazes_train.py [train args...]
"""
import os
import runpy
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if __name__ == "__main__":
    runpy.run_module("tasks.mazes.train", run_name="__main__")
