# Third-party notices

## HAL

The `HAL/hal/` directory contains the subset of [EQuS/HAL](https://github.com/EQuS/HAL) used for FoldQ's code definitions, placement helpers, multilayer routing, geometry, and hardware accounting. The generalized toric parameters and quantum Tanner parity-check matrices are distributed with that subset. Copyright (c) 2025 HAL contributors; licensed under MIT. The original notice is retained in `HAL/LICENSE`.

The local copy includes routing fixes in `routing/straightline.py`, `routing/astar.py`, and `routing/routing_algo.py`. The exact distributed files are identified by SHA-256 in `provenance/manifest.json`. The original upstream commit is unavailable because the source directory had no Git history.

`HAL/foldq/experiments/` contains FoldQ's integration and acceptance code, not upstream HAL code.

## Python packages

Dependencies are installed separately from their package distributions and retain their respective licenses. This repository does not vendor their implementations.
