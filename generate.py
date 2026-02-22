#!/usr/bin/env python3
"""Generate an airfoil from a trained model. Run: python generate.py --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from drl_airfoil.generate import main

if __name__ == "__main__":
    main()
