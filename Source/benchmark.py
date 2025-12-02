import time
import tracemalloc
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io
from typing import Dict, List, Any

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


def count_islands(grid: np.ndarray) -> int:
    """Helper to count islands."""
    return np.count_nonzero(grid > 0)


def run_benchmark():
    # Configuration
    input_dir = 'Inputs'
    input_files = sorted(glob.glob(os.path.join(input_dir, 'input_*.txt')))

    if not input_files:
        print(f"No input files found in '{input_dir}'!")
        return

    # Algorithm Configuration
    # Format: (Display Name, Function Object, Island Count Limit)
    algorithms = [
        # PySAT: No real limit needed
        ("PySAT", solve_with_pysat, 999),

        # Backtracking: Increased to 50 islands
        ("Backtracking", solve_backtracking, 50),

        # A*: Increased to 25 islands (Watch your RAM!)
        ("A* Search", solve_with_astar, 25),

        # Brute-force: Increased to 14 islands
        ("Brute-force", solve_bruteforce, 14)
    ]

    results = {algo[0]: {'times': [], 'mems': [], 'files': []} for algo in algorithms}
    file_labels = []

    # Header with fixed width formatting
    header = f"{'File':<15} | {'Islands':<7} | {'Algorithm':<15} | {'Time (s)':<10} | {'Mem (KB)':<10} | {'Status'}"
    print(header)
    print("-" * len(header))

    for file_path in input_files:
        filename = os.path.basename(file_path)
        file_labels.append(filename.replace(".txt", ""))

        try:
            grid = read_input_file(filename)
            num_islands = count_islands(grid)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        solver = HashiSolver(grid)

        for algo_name, func_object, island_limit in algorithms:
            # --- COMPLEXITY GUARDRAILS ---
            if num_islands > island_limit:
                results[algo_name]['times'].append(None)
                results[algo_name]['mems'].append(None)
                results[algo_name]['files'].append(filename)
                print(
                    f"{filename:<15} | {num_islands:<7} | {algo_name:<15} | {'SKIP':<10} | {'SKIP':<10} | > {island_limit} Islands")
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
                        # Increased time limit to 20s
                        solution = func_object(solver, time_limit=20.0)
                        if not solution and (time.perf_counter() - start_time > 19.9):
                            status = "Timeout"

                    elif algo_name == "Brute-force":
                        # Increased max pairs to 22
                        solution = func_object(solver, first_solution=True, max_pairs=22)
                        if not solution:
                            status = "Refused/Fail"

                    else:
                        solution = func_object(solver)

                    if not solution and status == "Success":
                        status = "No Solution"

                except Exception as e:
                    status = f"Err: {str(e)[:15]}..."

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

            print(
                f"{filename:<15} | {num_islands:<7} | {algo_name:<15} | {duration:.4f}     | {peak_mem_kb:.2f}       | {status}")

    plot_results(results, file_labels)


def plot_results(results, file_labels):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 1. Time Plot
    for algo_name, data in results.items():
        clean_times = [t if t is not None else np.nan for t in data['times']]
        ax1.plot(file_labels, clean_times, marker='o', label=algo_name, linewidth=2)

    ax1.set_title('Execution Time (Log Scale)')
    ax1.set_ylabel('Time (s)')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.legend()

    # 2. Memory Plot
    for algo_name, data in results.items():
        clean_mems = [m if m is not None else np.nan for m in data['mems']]
        ax2.plot(file_labels, clean_mems, marker='s', linestyle='--', label=algo_name)

    ax2.set_title('Peak Memory Usage')
    ax2.set_ylabel('Memory (KB)')
    ax2.set_xlabel('Test Cases')
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    print("\nDisplaying plot...")
    plt.show()


if __name__ == "__main__":
    run_benchmark()