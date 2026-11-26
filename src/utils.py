"""
utils.py - Utility helpers: seed setting, timer, printing.
"""
import random
import numpy as np
import torch
import time


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Timer:
    """Simple context manager to time blocks of code."""
    def __init__(self, label=""):
        self.label = label
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.label:
            print(f"  [{self.label}] {self.elapsed:.1f}s")


def print_table(headers, rows, title=""):
    """Print a simple text table to the console."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    # Print header
    hdr = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(hdr)
    print("-+-".join("-" * w for w in col_widths))
    # Print rows
    for row in rows:
        line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(line)
    print()
