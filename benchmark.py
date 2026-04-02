"""
benchmark.py — Head-to-head: C++ Pybind11 parser vs pure Python baseline.

Measures:
  - Wall-clock throughput (lines/sec, MB/sec)
  - Speedup ratio
  - CPU utilization delta (via psutil sampling)
  - Output correctness (both parsers must agree on all fields)

Usage:
    python benchmark.py                    # default: 500k lines, 5 warm-up runs
    python benchmark.py --lines 1000000    # 1M lines
    python benchmark.py --runs 10          # more iterations for stable numbers
    python benchmark.py --no-cpu           # skip CPU measurement (faster)
"""

import argparse
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--lines",  type=int,  default=500_000)
parser.add_argument("--runs",   type=int,  default=5)
parser.add_argument("--no-cpu", action="store_true")
parser.add_argument("--log",    type=str,  default="logs/telemetry.log")
args = parser.parse_args()

# ── Imports (after build check) ───────────────────────────────────────────────
try:
    import logparser_cpp as cpp_parser
except ImportError:
    print("ERROR: C++ extension not built. Run: pip install -e .")
    sys.exit(1)

from logparser_py import parse_log_file as py_parser

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    if not args.no_cpu:
        print("Note: psutil not installed — skipping CPU measurement. "
              "Install with: pip install psutil")

# ── Generate log if missing ───────────────────────────────────────────────────
log_path = args.log
if not Path(log_path).exists():
    print(f"Log file not found. Generating {args.lines:,} lines...")
    from generate_logs import generate_logs
    generate_logs(args.lines, log_path)

file_size_mb = Path(log_path).stat().st_size / (1024 * 1024)
actual_lines = sum(1 for _ in open(log_path))
print(f"\nLog: {log_path}  ({actual_lines:,} lines, {file_size_mb:.1f} MB)\n")

# ── CPU sampler ───────────────────────────────────────────────────────────────
class CPUSampler:
    """Samples CPU utilization in a background thread."""
    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples  = []
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        return statistics.mean(self.samples) if self.samples else 0.0

    def _run(self):
        proc = psutil.Process()
        while not self._stop.is_set():
            self.samples.append(proc.cpu_percent(interval=None))
            time.sleep(self.interval)

# ── Benchmark runner ──────────────────────────────────────────────────────────
def run_benchmark(fn, label, n_runs, measure_cpu):
    times = []
    cpu_pcts = []

    for i in range(n_runs):
        sampler = CPUSampler().start() if (measure_cpu and HAS_PSUTIL) else None
        t0      = time.perf_counter()
        result  = fn(log_path)
        elapsed = time.perf_counter() - t0
        cpu     = sampler.stop() if sampler else None

        times.append(elapsed)
        if cpu is not None:
            cpu_pcts.append(cpu)

        status = f"{elapsed:.3f}s"
        if cpu is not None:
            status += f"  CPU {cpu:.1f}%"
        print(f"  {label} run {i+1}/{n_runs}: {status}")

    return times, cpu_pcts, result

# ── Run ───────────────────────────────────────────────────────────────────────
measure_cpu = HAS_PSUTIL and not args.no_cpu

print("=" * 60)
print("  PURE PYTHON (baseline)")
print("=" * 60)
py_times, py_cpus, py_result = run_benchmark(
    py_parser, "Python", args.runs, measure_cpu)

print()
print("=" * 60)
print("  C++ / PYBIND11")
print("=" * 60)
cpp_times, cpp_cpus, cpp_result = run_benchmark(
    cpp_parser.parse_log_file, "C++   ", args.runs, measure_cpu)

# ── Correctness check ─────────────────────────────────────────────────────────
print("\n── Correctness ──────────────────────────────────────────")
assert len(py_result)  == len(cpp_result),  \
    f"Row count mismatch: Python={len(py_result)} C++={len(cpp_result)}"
assert np.allclose(py_result["cpu"],     cpp_result["cpu"],     atol=1e-6), "cpu mismatch"
assert np.array_equal(py_result["mem"],  cpp_result["mem"]),                "mem mismatch"
assert np.allclose(py_result["latency"], cpp_result["latency"], atol=1e-6), "latency mismatch"
assert np.array_equal(py_result["status"], cpp_result["status"]),           "status mismatch"
assert np.array_equal(py_result["level"],  cpp_result["level"]),            "level mismatch"
print(f"  ✓ Both parsers agree on all {len(py_result):,} records.")

# ── Summary ───────────────────────────────────────────────────────────────────
py_med  = statistics.median(py_times)
cpp_med = statistics.median(cpp_times)
speedup = py_med / cpp_med

py_tput  = actual_lines / py_med  / 1e6
cpp_tput = actual_lines / cpp_med / 1e6

py_mb  = file_size_mb / py_med
cpp_mb = file_size_mb / cpp_med

print("\n── Results ──────────────────────────────────────────────")
print(f"  {'':20}  {'Python':>12}  {'C++':>12}  {'Δ':>10}")
print(f"  {'Median time (s)':20}  {py_med:>12.3f}  {cpp_med:>12.3f}  {speedup:>9.2f}x")
print(f"  {'Throughput (M lines/s)':20}  {py_tput:>12.2f}  {cpp_tput:>12.2f}")
print(f"  {'Throughput (MB/s)':20}  {py_mb:>12.1f}  {cpp_mb:>12.1f}")

if measure_cpu and py_cpus and cpp_cpus:
    py_cpu  = statistics.mean(py_cpus)
    cpp_cpu = statistics.mean(cpp_cpus)
    cpu_reduction = (py_cpu - cpp_cpu) / py_cpu * 100 if py_cpu > 0 else 0
    print(f"  {'Avg CPU util (%)':20}  {py_cpu:>12.1f}  {cpp_cpu:>12.1f}  "
          f"{cpu_reduction:>+8.1f}%")

print(f"\n  Speedup: {speedup:.2f}x  "
      f"(median over {args.runs} runs, {actual_lines:,} lines)")
print()
