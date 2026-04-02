"""
setup.py — builds the Pybind11 C++ extension.

Usage:
    pip install .              # build + install
    pip install -e .           # editable / development install
    python setup.py build_ext --inplace   # build .so in-place for quick testing
"""

from setuptools import setup, Extension
import pybind11
import sys

extra_compile_args = [
    "-O3",           # maximum optimization
    "-std=c++17",    # from_chars requires C++17
    "-march=native", # use all CPU features available on build machine
    "-ffast-math",   # aggressive float optimization (safe for our use case)
    "-Wall",
]

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17", "/W3"]

ext = Extension(
    name="logparser_cpp",
    sources=["src/logparser.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=extra_compile_args,
    language="c++",
)

setup(
    name="logparser",
    version="1.0.0",
    description="High-throughput C++ telemetry log parser via Pybind11",
    ext_modules=[ext],
    python_requires=">=3.9",
    install_requires=["pybind11>=2.11", "numpy>=1.24"],
)
