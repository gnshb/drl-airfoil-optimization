#!/usr/bin/env python3
"""Train a DRL airfoil optimizer. Run: python train.py --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from drl_airfoil.train import main

if __name__ == "__main__":
    main()
