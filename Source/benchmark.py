import time
import tracemalloc
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io
from typing import Dict, List, Any, Tuple, Set

# Import class and standalone algorithms from your main script
try:
    from main import (
        HashiSolver,
        read_input_file,
        solve_with_pysat,
        solve_with_astar,
        solve_backtracking,
        solve_bruteforce
    )
except ImportError:
    print("Error: main.py not found or missing specific functions.")
    print("Ensure benchmark_hashi.py is in the same folder as the refactored main.py")
    exit()


def count_potential_bridges(solver: HashiSolver) -> int:
    """
    Counts the total number of unique potential bridges (pairs of connectable islands).
    This (M) is the true complexity factor, not the number of islands (N).
    Complexity is roughly O(3^M).
    """
    unique_pairs = set()

    # Iterate through all islands
    for island in solver.islands:
        i, j, _ = island
        # find_neighbors expects a tuple (r, c, req) based on the provided class
        neighbors = solver.find_neighbors(island)

        for ni, nj in neighbors:
            # Create a normalized pair (min, max) to avoid duplicates like (A,B) and (B,A)
            p1 = (i, j)
            p2 = (ni, nj)
            pair = tuple(sorted((p1, p2)))
            unique_pairs.add(pair)

    return len(unique_pairs)


def run_benchmark():
    # Configuration
    input_dir = 'Inputs'
    # Sort by filename naturally
    input_files = sorted(glob.glob(os.path.join(input_dir, 'input_*.txt')))

    if not input_files:
        print(f"No input files found in '{input_dir}'!")
        return

    # Algorithm Configuration
    # Format: (Display Name, Function Object, POTENTIAL BRIDGE (PAIR) LIMIT)
    algorithms = [
        # PySAT: SAT solvers handle clauses well, can handle high complexity
        ("PySAT", solve_with_pysat, 9999999999),

        # Backtracking: Optimized with pruning, can handle medium complexity (~50-60 pairs)
        ("Backtracking", solve_backtracking, 70),

        # A*: Memory bound. State space explodes fast. Limit strictly.
        ("A* Search", solve_with_astar, 40),

        # Brute-force: Pure O(3^M). 3^15 is ~14 million. 3^18 is ~387 million.
        # Strict limit to prevent hanging.
        ("Brute-force", solve_bruteforce, 16)
    ]

    results = {algo[0]: {'times': [], 'mems': [], 'files': [], 'pairs': []} for algo in algorithms}
    file_labels = []

    # Header with fixed width formatting
    header = f"{'File':<15} | {'Islands':<7} | {'Pairs(M)':<8} | {'Algorithm':<15} | {'Time (s)':<10} | {'Mem (KB)':<10} | {'Status'}"
    print(header)
    print("-" * len(header))

    for file_path in input_files:
        filename = os.path.basename(file_path)

        try:
            grid = read_input_file(filename)
            # Create solver instance immediately to calculate neighbors
            solver = HashiSolver(grid)
            num_islands = len(solver.islands)
            num_pairs = count_potential_bridges(solver)

            # Label for graph: "input-01 (M=12)"
            file_labels.append(f"{filename.replace('.txt', '')}\n(M={num_pairs})")

        except Exception as e:
            print(f"Error preparing {filename}: {e}")
            continue

        for algo_name, func_object, pair_limit in algorithms:
            # --- COMPLEXITY GUARDRAILS BASED ON PAIRS (M) ---
            if num_pairs > pair_limit:
                results[algo_name]['times'].append(None)
                results[algo_name]['mems'].append(None)
                results[algo_name]['files'].append(filename)
                results[algo_name]['pairs'].append(num_pairs)
                print(
                    f"{filename:<15} | {num_islands:<7} | {num_pairs:<8} | {algo_name:<15} | {'SKIP':<10} | {'SKIP':<10} | > {pair_limit} Pairs")
                continue

            # --- BENCHMARKING ---
            tracemalloc.start()
            start_time = time.perf_counter()

            status = "Success"
            solution = {}

            # Use redirect_stdout to suppress internal prints from the algorithms
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    if algo_name == "Backtracking":
                        solution = func_object(solver, time_limit=20.0)
                        if not solution and (time.perf_counter() - start_time > 19.9):
                            status = "Timeout"

                    elif algo_name == "Brute-force":
                        # Ensure brute force respects the same limit internally if possible
                        solution = func_object(solver, first_solution=True, max_pairs=pair_limit + 2)
                        if not solution:
                            status = "Refused/Fail"

                    else:
                        solution = func_object(solver)

                    if not solution and "Fail" not in status and "Timeout" not in status:
                        status = "No Solution"

                except Exception as e:
                    status = f"Err: {str(e)[:10]}.."

            end_time = time.perf_counter()
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Process metrics
            duration = end_time - start_time
            peak_mem_kb = peak_mem / 1024

            # Filter out brute-force refusals (too fast to be a real run)
            if algo_name == "Brute-force" and duration < 0.001 and not solution:
                results[algo_name]['times'].append(None)
                results[algo_name]['mems'].append(None)
            else:
                results[algo_name]['times'].append(duration)
                results[algo_name]['mems'].append(peak_mem_kb)

            results[algo_name]['files'].append(filename)
            results[algo_name]['pairs'].append(num_pairs)

            print(
                f"{filename:<15} | {num_islands:<7} | {num_pairs:<8} | {algo_name:<15} | {duration:.4f}     | {peak_mem_kb:.2f}       | {status}")

    plot_results(results, file_labels)


def plot_results(results, file_labels):
    # Only pick unique file labels (since loops run multiple times per file)
    # The file_labels list grew inside the file loop, but we only need unique entries for X-axis
    unique_labels = []
    seen = set()
    for l in file_labels:
        if l not in seen:
            unique_labels.append(l)
            seen.add(l)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Time Plot
    for algo_name, data in results.items():
        # Ensure data aligns with unique labels (handle skips)
        clean_times = [t if t is not None else np.nan for t in data['times']]

        # If files were skipped entirely before loop, lengths might mismatch.
        # But here we append None or Value for every file in the loop.

        ax1.plot(unique_labels, clean_times, marker='o', label=algo_name, linewidth=2)

    ax1.set_title('Execution Time (Log Scale) vs Complexity (M=Pairs)')
    ax1.set_ylabel('Time (s)')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.legend()

    # 2. Memory Plot
    for algo_name, data in results.items():
        clean_mems = [m if m is not None else np.nan for m in data['mems']]
        ax2.plot(unique_labels, clean_mems, marker='s', linestyle='--', label=algo_name)

    ax2.set_title('Peak Memory Usage vs Complexity')
    ax2.set_ylabel('Memory (KB)')
    ax2.set_xlabel('Test Cases (M = Potential Bridges)')
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    print("\nDisplaying plot...")
    plt.show()


if __name__ == "__main__":
    run_benchmark()