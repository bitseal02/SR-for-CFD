"""Shared run/output utility functions for BFS workflows."""

import os
from datetime import datetime
from typing import Dict


def create_timestamped_output_dir(base_dir: str = "outputs") -> str:
    """Create and return an output directory named with current timestamp."""
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_run_summary(filepath: str, info: Dict[str, Dict[str, str]]):
    """Save simulation configuration and run stats as plain text."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{'='*70}\n")
            f.write("BFS SIMULATION RUN SUMMARY\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")

            for section, content in info.items():
                f.write(f"{section.upper()}\n")
                f.write(f"{'-'*len(section)}\n")
                for key, value in content.items():
                    f.write(f"{key:<30}: {value}\n")
                f.write("\n")

        print(f"Run summary saved to: {filepath}")
    except Exception as e:
        print(f"Failed to save run summary: {e}")
