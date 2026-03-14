# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
./jomake   # builds: cd build && cmake .. && make
./jorun    # runs: build/run
```

C++17, CMake. Single executable `run` built from `src/main.cpp`.

## Architecture

JojoGrad is a reverse-mode automatic differentiation library in C++.

**Core:** `include/jojograd/variable.hpp`
- `variable<T>` — template node in a DAG; holds `data`, `grad`, `children` (up to 2 parent pointers), and `op` token
- `operation` — dispatches op tokens (0=add, 1=multiply, 2=subtract, 3=divide) to gradient formulas
- Operator overloads (`+`, `-`, `*`, `/`, unary `-`) create new `variable` nodes linked to their operands
- `backward(gradient)` — recursively applies the chain rule up the graph, accumulating gradients with `+=`

**Entry point:** `src/main.cpp` — usage examples/manual tests.

**`include/trash/`** — incomplete experimental code, not part of the build.
