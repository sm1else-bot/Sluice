"""
generate_logs.py — Synthetic telemetry log generator.

Generates realistic-looking telemetry logs for benchmarking and testing.

Usage:
    python generate_logs.py                  # 500k lines → logs/telemetry.log
    python generate_logs.py --lines 1000000  # 1M lines
    python generate_logs.py --lines 100000 --out logs/small.log
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta

LEVELS   = ["DEBUG", "INFO", "INFO", "INFO", "WARN", "ERROR", "FATAL"]  # weighted
NODES    = [f"node-{i:02d}" for i in range(1, 33)]  # 32 nodes

def generate_logs(n_lines: int, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    ts   = datetime(2024, 1, 15, 0, 0, 0)
    delta = timedelta(microseconds=500)  # 2000 events/sec

    rng = random.Random(42)  # deterministic

    with open(out_path, "w", buffering=1 << 20) as f:  # 1MB write buffer
        for _ in range(n_lines):
            ts += delta
            level   = rng.choice(LEVELS)
            node    = rng.choice(NODES)
            cpu     = round(rng.uniform(0.5, 99.5), 2)
            mem     = rng.randint(256, 65536)
            latency = round(rng.uniform(0.1, 500.0), 2)
            status  = rng.choices([0, 1, 2, 3], weights=[85, 8, 5, 2])[0]

            f.write(
                f"{ts.strftime('%Y-%m-%dT%H:%M:%S.%f')} "
                f"{level:<5} {node} "
                f"cpu={cpu} mem={mem} latency={latency} status={status}\n"
            )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Generated {n_lines:,} lines → {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=500_000)
    parser.add_argument("--out",   type=str, default="logs/telemetry.log")
    args = parser.parse_args()
    generate_logs(args.lines, args.out)
