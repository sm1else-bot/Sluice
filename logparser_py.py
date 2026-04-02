"""
logparser_py.py — Pure Python baseline parser.

This is what the C++ extension replaces. Intentionally written the way
most engineers would naturally write it — no exotic micro-optimizations,
but not deliberately slow either. A fair baseline.
"""

import re
from pathlib import Path

import numpy as np

LEVEL_MAP = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "FATAL": 4}

# Pre-compiled regex for the log format:
# <timestamp> <LEVEL> <node> cpu=<f> mem=<i> latency=<f> status=<i>
_PATTERN = re.compile(
    r"^\S+\s+"                       # timestamp
    r"(\w+)\s+"                      # level
    r"\S+\s+"                        # node_id
    r"cpu=([\d.]+)\s+"               # cpu
    r"mem=(\d+)\s+"                  # mem
    r"latency=([\d.]+)\s+"           # latency
    r"status=(\d+)"                  # status
)


def parse_log_file(path: str, skip_malformed: bool = True) -> np.ndarray:
    """
    Parse a telemetry log file and return a structured NumPy array.

    Args:
        path:           Path to the log file.
        skip_malformed: Skip lines that don't match the expected format.

    Returns:
        np.ndarray with dtype:
        [('level','<i4'), ('cpu','<f8'), ('mem','<i8'), ('latency','<f8'), ('status','<i4')]
    """
    dtype = np.dtype([
        ("level",   np.int32),
        ("cpu",     np.float64),
        ("mem",     np.int64),
        ("latency", np.float64),
        ("status",  np.int32),
    ])

    records = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n\r")
            if not line:
                continue
            m = _PATTERN.match(line)
            if not m:
                if not skip_malformed:
                    raise ValueError(f"Malformed line {lineno}: {line!r}")
                continue

            level_str, cpu_s, mem_s, lat_s, stat_s = m.groups()
            records.append((
                LEVEL_MAP.get(level_str, -1),
                float(cpu_s),
                int(mem_s),
                float(lat_s),
                int(stat_s),
            ))

    if not records:
        return np.empty(0, dtype=dtype)

    return np.array(records, dtype=dtype)
