# Copilot Instructions for Sokoban Solver Codebase

## Overview
This project is a high-performance Sokoban solver written in C++. It features multiple algorithmic variants and optimizations, with the main workflow centered around `hw1.cpp` and related source files. The codebase is designed for competitive programming and research on state-space search and compression.

## Architecture & Key Files
- **`hw1.cpp`**: Main entry point and core solver logic. Implements A* and IDA* variants, state compression, and deadlock detection.
- **`Makefile`**: Defines build targets for different solver variants. Use `make` to build the default version, or specify targets for optimized versions.
- **`samples/`**: Contains Sokoban level files for testing.
- **`test.sh`, `detailed_test.sh`**: Shell scripts for batch testing and validation.
- **`validate.py`**: Python script to check solution validity.
- **`optimization_suggestions.md`, `state_compression_analysis.md`**: Documentation of optimization ideas and compression techniques.

## Developer Workflows
- **Build**: Run `make` in the project root. Use the Makefile to select different solver variants (e.g., `make hw1_optimized`).
- **Test**: Use `test.sh` or `detailed_test.sh` to run the solver on all sample levels. Example: `./hw1 ./samples/01.txt`.
- **Validate**: After solving, use `validate.py` to check the correctness of the output.
- **Debug**: For algorithmic debugging, focus on `hw1.cpp` and its variants. Use print statements or logging for state inspection.

## Project-Specific Patterns
- **State Representation**: Uses both `GameState` (map-based) and `BitState` (bitset-based) for performance. Most optimized flows use `BitState` exclusively.
- **Deadlock Detection**: Custom logic in `is_deadlock` and related functions. Avoid generic deadlock code; follow the project’s patterns.
- **Heuristic Calculation**: See `calculate_heuristic` and its variants. Patterns include greedy matching and pattern database heuristics.
- **Compression**: State compression is critical. See `init_compression` and related analysis docs.
- **Multiple Solver Variants**: Each `.cpp` file (e.g., `hw1_optimized.cpp`, `hw1_faster.cpp`) may implement a different algorithm or optimization. Refer to the Makefile for build options.

## Conventions
- Prefer bitset/BitState for new optimizations.
- Use the provided scripts for batch testing and validation.
- Keep all new solver variants in separate `.cpp` files for easy benchmarking.
- Document new optimization ideas in `optimization_suggestions.md`.

## Integration Points
- No external dependencies beyond standard C++ and Python 3 for validation.
- Scripts and Makefile are tightly coupled to the file naming conventions.

---

For further details, see comments in `hw1.cpp` and the markdown documentation files. If you add new solver variants or optimizations, update this file and the Makefile accordingly.
