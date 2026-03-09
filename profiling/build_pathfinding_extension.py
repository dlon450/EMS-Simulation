#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, setup
from Cython.Build import cythonize

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

extensions = [
    Extension(
        name="src.pathfinding_cy",
        sources=[str(ROOT / "src" / "pathfinding_cy.pyx")],
    )
]

setup(
    name="ems-simulation-pathfinding-cython",
    ext_modules=cythonize(
        extensions,
        language_level="3",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
        annotate=False,
    ),
    script_args=["build_ext", "--inplace"],
)
