import numpy as np
from typing import List, Tuple, Dict, Set
from pysat.solvers import Glucose3
import heapq
import os
import math
import time
import itertools


class HashiSolver:
    """
    Model class for the Hashiwokakero puzzle.
    Handles grid storage, island detection, and geometric rules (neighbors, crossing).
    """

    def __init__(self, grid: np.ndarray):
        """
        Initialize Hashiwokakero solver
        grid: 2D numpy array where 0 = empty, positive numbers = islands with required bridges
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.islands = self._find_islands()

    def _find_islands(self) -> List[Tuple[int, int, int]]:
        """Find all islands (non-zero cells) with their coordinates and required bridges"""
        islands = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] > 0:
                    islands.append((i, j, int(self.grid[i, j])))
        return islands

    def find_neighbors(self, island: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find all islands that can be connected to the given island"""
        i, j, _ = island
        neighbors = []

        # Check horizontal (right)
        for col in range(j + 1, self.cols):
            if self.grid[i, col] > 0:
                neighbors.append((i, col))
                break
            elif self.grid[i, col] < 0:  # Obstacle
                break

        # Check horizontal (left)
        for col in range(j - 1, -1, -1):
            if self.grid[i, col] > 0:
                neighbors.append((i, col))
                break
            elif self.grid[i, col] < 0:
                break

        # Check vertical (down)
        for row in range(i + 1, self.rows):
            if self.grid[row, j] > 0:
                neighbors.append((row, j))
                break
            elif self.grid[row, j] < 0:
                break

        # Check vertical (up)
        for row in range(i - 1, -1, -1):
            if self.grid[row, j] > 0:
                neighbors.append((row, j))
                break
            elif self.grid[row, j] < 0:
                break

        return neighbors

    def bridges_cross(self, bridge1: Tuple[Tuple[int, int], Tuple[int, int]],
                      bridge2: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """Check if two bridges cross each other"""
        (i1, j1), (i2, j2) = bridge1
        (i3, j3), (i4, j4) = bridge2

        # Check if one is horizontal and other is vertical
        bridge1_horizontal = i1 == i2
        bridge2_horizontal = i3 == i4

        if bridge1_horizontal == bridge2_horizontal:
            return False  # Both horizontal or both vertical - can't cross

        if bridge1_horizontal:
            # Bridge1 horizontal, Bridge2 vertical
            return (min(j1, j2) < j3 < max(j1, j2) and
                    min(i3, i4) < i1 < max(i3, i4))
        else:
            # Bridge1 vertical, Bridge2 horizontal
            return (min(i1, i2) < i3 < max(i1, i2) and
                    min(j3, j4) < j1 < max(j3, j4))

    def format_output(self, solution: Dict) -> List[List[str]]:
        """
        Format solution as output grid with bridge symbols.
        """
        # Initialize output grid with empty "0"
        output = [['0' for _ in range(self.cols)] for _ in range(self.rows)]

        # Place islands
        for i, j, req in self.islands:
            output[i][j] = str(req)

        # Place bridges
        for key, value in solution.items():
            if not value:
                continue

            # Normalize (key, value) → island1, island2, bridge_count
            island1 = island2 = None
            bridge_count = value

            try:
                # Case: (island1, island2, bridge_count)
                if isinstance(key, tuple) and len(key) == 3:
                    island1, island2, bridge_count = key

                # Case: ((island1, island2), bridge_count)
                elif isinstance(key, tuple) and len(key) == 2:
                    pair, bc = key
                    if isinstance(pair, tuple) and len(pair) == 2:
                        island1, island2 = pair
                        bridge_count = bc
                    else:
                        continue
                else:
                    continue
            except:
                continue

            if island1 is None or island2 is None:
                continue

            i1, j1 = island1
            i2, j2 = island2

            # Horizontal
            if i1 == i2:
                symbol = "=" if bridge_count == 2 else "-"
                for c in range(min(j1, j2) + 1, max(j1, j2)):
                    if output[i1][c] == "0":
                        output[i1][c] = symbol

            # Vertical
            elif j1 == j2:
                symbol = "$" if bridge_count == 2 else "|"
                for r in range(min(i1, i2) + 1, max(i1, i2)):
                    if output[r][j1] == "0":
                        output[r][j1] = symbol

            # Skip invalid diagonals
            else:
                continue

        return output


###############################################################################
# INDEPENDENT ALGORITHMS
###############################################################################

def solve_with_pysat(solver: HashiSolver) -> Dict[Tuple, int]:
    """
    Solve using PySAT SAT solver.
    Encapsulates CNF generation specific logic locally.
    """
    var_counter = 1
    var_map = {}  # Maps (island1, island2, bridge_type) to variable number
    reverse_var_map = {}

    def get_variable(island1, island2, bridge_count):
        nonlocal var_counter
        if island1 > island2:
            island1, island2 = island2, island1
        key = (island1, island2, bridge_count)
        if key not in var_map:
            var_map[key] = var_counter
            reverse_var_map[var_counter] = key
            var_counter += 1
        return var_map[key]

    def exactly_n_bridges(bridge_vars, n):
        clauses = []
        from itertools import product
        k = len(bridge_vars)
        for assignment in product([0, 1, 2], repeat=k):
            total = sum(assignment)
            if total == n:
                continue
            clause = []
            for idx, state in enumerate(assignment):
                s_var, d_var = bridge_vars[idx]
                if state == 0:
                    clause.append(s_var);
                    clause.append(d_var)
                elif state == 1:
                    clause.append(-s_var);
                    clause.append(d_var)
                elif state == 2:
                    clause.append(s_var);
                    clause.append(-d_var)
            clauses.append(clause)
        return clauses

    def no_crossing_bridges():
        clauses = []
        bridge_pairs = list(var_map.keys())
        for i in range(len(bridge_pairs)):
            for j in range(i + 1, len(bridge_pairs)):
                bridge1 = bridge_pairs[i]
                bridge2 = bridge_pairs[j]
                if solver.bridges_cross(bridge1[:2], bridge2[:2]):
                    var1 = var_map[bridge1]
                    var2 = var_map[bridge2]
                    clauses.append([-var1, -var2])
        return clauses

    # Generate CNF
    clauses = []
    for island in solver.islands:
        i, j, required = island
        neighbors = solver.find_neighbors(island)
        if not neighbors:
            continue
        bridge_vars = []
        for neighbor in neighbors:
            var1 = get_variable((i, j), neighbor, 1)
            var2 = get_variable((i, j), neighbor, 2)
            bridge_vars.append((var1, var2))
        clauses.extend(exactly_n_bridges(bridge_vars, required))
        for var1, var2 in bridge_vars:
            clauses.append([-var1, -var2])

    clauses.extend(no_crossing_bridges())

    if not clauses:
        print("No clauses generated!")
        return {}

    sat_solver = Glucose3()
    for clause in clauses:
        sat_solver.add_clause(clause)

    if sat_solver.solve():
        model = sat_solver.get_model()
        solution = {}
        for var in model:
            if var > 0 and var in reverse_var_map:
                key = reverse_var_map[var]
                solution[key] = 1
        return solution
    else:
        print("No solution found with SAT solver")
        return {}


def solve_with_astar(solver: HashiSolver) -> Dict[Tuple, int]:
    """Solve using A* search algorithm."""

    class State:
        def __init__(self, bridges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int],
                     remaining: Dict[Tuple[int, int], int]):
            self.bridges = dict(bridges)
            self.remaining = dict(remaining)
            self.cost = sum(self.bridges.values())
            self.heuristic = 0

        def f(self):
            return self.cost + self.heuristic

        def __lt__(self, other):
            return self.f() < other.f()

    def pair_key(a, b):
        return (a, b) if a <= b else (b, a)

    def segments_overlap(a1, a2, b1, b2):
        (i1, j1), (i2, j2) = a1, a2
        (i3, j3), (i4, j4) = b1, b2
        if i1 == i2 == i3 == i4:
            a_min, a_max = sorted((j1, j2))
            b_min, b_max = sorted((j3, j4))
            return not (a_max <= b_min or b_max <= a_min)
        if j1 == j2 == j3 == j4:
            a_min, a_max = sorted((i1, i2))
            b_min, b_max = sorted((i3, i4))
            return not (a_max <= b_min or b_max <= a_min)
        return False

    def is_valid_placement(a, b, new_count, existing_bridges):
        (i1, j1), (i2, j2) = a, b
        if not (i1 == i2 or j1 == j2): return False
        if new_count > 2 or new_count < 0: return False

        for (c1, c2), cnt in existing_bridges.items():
            if cnt <= 0: continue
            if pair_key(c1, c2) == pair_key(a, b): continue

            try:
                crosses = solver.bridges_cross((c1, c2), (a, b))
            except Exception:
                crosses = False

            if crosses: return False

            if (c1[0] == c2[0] == a[0] == b[0]) or (c1[1] == c2[1] == a[1] == b[1]):
                if segments_overlap(c1, c2, a, b): return False
        return True

    def count_components(bridges_map):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x: parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra

        for (i, j, _) in solver.islands:
            parent[(i, j)] = (i, j)
        for (a, b), cnt in bridges_map.items():
            if cnt > 0: union(a, b)
        return len(set(find(x) for x in parent))

    def compute_heuristic(bridges_map, remaining_map):
        total_remaining = sum(remaining_map.values())
        min_segments = math.ceil(total_remaining / 2)
        components = count_components(bridges_map)
        needed_to_connect = max(0, components - 1)
        return max(min_segments, needed_to_connect)

    initial_remaining = {(i, j): req for i, j, req in solver.islands}
    initial_bridges = {}
    start = State(initial_bridges, initial_remaining)
    start.heuristic = compute_heuristic(start.bridges, start.remaining)

    open_set = []
    heapq.heappush(open_set, (start.f(), id(start), start))
    closed = set()
    iterations = 0
    max_iterations = 200000

    while open_set and iterations < max_iterations:
        iterations += 1
        _, _, cur = heapq.heappop(open_set)

        if all(v == 0 for v in cur.remaining.values()):
            if count_components(cur.bridges) == 1:
                solution = {}
                for (a, b), cnt in cur.bridges.items():
                    if cnt > 0:
                        a_key, b_key = pair_key(a, b)
                        solution[(a_key, b_key, cnt)] = 1
                return solution

        state_key = (tuple(sorted(((k[0], k[1]), v) for k, v in cur.bridges.items())),
                     tuple(sorted(cur.remaining.items())))
        if state_key in closed: continue
        closed.add(state_key)

        islands_with_rem = [(rem, isl) for isl, rem in cur.remaining.items() if rem > 0]
        if not islands_with_rem: continue
        _, expand_island = max(islands_with_rem)

        i, j = expand_island
        neighbors = solver.find_neighbors((i, j, 0))
        for ni, nj in neighbors:
            if cur.remaining.get((ni, nj), 0) <= 0: continue
            pair = pair_key((i, j), (ni, nj))
            current_count = cur.bridges.get(pair, 0)

            for add in (1, 2):
                new_count = current_count + add
                if new_count > 2: continue
                if cur.remaining[(i, j)] < add or cur.remaining[(ni, nj)] < add: continue
                if not is_valid_placement(pair[0], pair[1], new_count, cur.bridges): continue

                new_bridges = dict(cur.bridges)
                new_bridges[pair] = new_count
                new_remaining = dict(cur.remaining)
                new_remaining[(i, j)] -= add
                new_remaining[(ni, nj)] -= add

                new_state = State(new_bridges, new_remaining)
                new_state.heuristic = compute_heuristic(new_state.bridges, new_state.remaining)
                heapq.heappush(open_set, (new_state.f(), id(new_state), new_state))

    print(f"A* stopped after {iterations} iterations without finding a connected solution")
    return {}


def solve_bruteforce(solver: HashiSolver, first_solution: bool = True, max_pairs: int = 18) -> Dict[Tuple, int]:
    """Brute-force solver independent function."""
    start = time.time()
    pairs = []
    seen = set()
    for i, j, _ in solver.islands:
        neighbors = solver.find_neighbors((i, j, 0))
        for ni, nj in neighbors:
            a, b = (i, j), (ni, nj)
            key = (a, b) if a <= b else (b, a)
            if key not in seen:
                seen.add(key)
                pairs.append(key)

    m = len(pairs)
    if m == 0:
        print("No pairs to connect.")
        return {}
    if m > max_pairs:
        print(f"Brute-force refused: too many pairs ({m})")
        return {}

    def segments_overlap(a1, a2, b1, b2):
        (i1, j1), (i2, j2) = a1, a2
        (i3, j3), (i4, j4) = b1, b2
        if i1 == i2 == i3 == i4:
            a_min, a_max = sorted((j1, j2))
            b_min, b_max = sorted((j3, j4))
            return not (a_max <= b_min or b_max <= a_min)
        if j1 == j2 == j3 == j4:
            a_min, a_max = sorted((i1, i2))
            b_min, b_max = sorted((i3, i4))
            return not (a_max <= b_min or b_max <= a_min)
        return False

    islands_req = {(i, j): req for i, j, req in solver.islands}
    island_list = list(islands_req.keys())
    total_checked = 0

    for assignment in itertools.product([0, 1, 2], repeat=m):
        total_checked += 1
        sums = {k: 0 for k in islands_req}
        valid = True
        for idx, cnt in enumerate(assignment):
            if cnt == 0: continue
            a, b = pairs[idx]
            sums[a] += cnt
            sums[b] += cnt
            if sums[a] > islands_req[a] or sums[b] > islands_req[b]:
                valid = False
                break
        if not valid: continue
        if any(sums[k] != islands_req[k] for k in islands_req): continue

        placed = [(pairs[idx], assignment[idx]) for idx in range(m) if assignment[idx] > 0]
        fail = False
        for i1 in range(len(placed)):
            (a1, b1), c1 = placed[i1]
            for i2 in range(i1 + 1, len(placed)):
                (a2, b2), c2 = placed[i2]
                try:
                    if solver.bridges_cross((a1, b1), (a2, b2)):
                        fail = True;
                        break
                except Exception:
                    # Fallback manual check
                    (r1c1, c1c1), (r1c2, c1c2) = a1, b1
                    (r2c1, c2c1), (r2c2, c2c2) = a2, b2
                    if (r1c1 == r1c2 and c2c1 == c2c2):
                        if (min(c1c1, c1c2) < c2c1 < max(c1c1, c1c2) and
                                min(r2c1, r2c2) < r1c1 < max(r2c1, r2c2)):
                            fail = True;
                            break
                    if (c1c1 == c1c2 and r2c1 == r2c2):
                        if (min(r1c1, r1c2) < r2c1 < max(r1c1, r1c2) and
                                min(c2c1, c2c2) < c1c1 < max(c2c1, c2c2)):
                            fail = True;
                            break

                if (a1[0] == b1[0] == a2[0] == b2[0]) or (a1[1] == b1[1] == a2[1] == b2[1]):
                    if segments_overlap(a1, b1, a2, b2):
                        fail = True;
                        break
            if fail: break
        if fail: continue

        parent = {}

        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x: parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry: parent[ry] = rx

        for isl in island_list: parent[isl] = isl
        for (a, b), cnt in placed:
            if cnt > 0: union(a, b)
        roots = set(find(x) for x in island_list)
        if len(roots) != 1: continue

        solution = {}
        for idx, cnt in enumerate(assignment):
            if cnt > 0:
                a, b = pairs[idx]
                a_key, b_key = (a, b) if a <= b else (b, a)
                solution[(a_key, b_key, cnt)] = 1

        elapsed = time.time() - start
        print(f"Brute-force: found solution after checking {total_checked} assignments in {elapsed:.3f}s")
        if first_solution: return solution

    elapsed = time.time() - start
    print(f"Brute-force: finished checking {total_checked} assignments in {elapsed:.3f}s — no solution")
    return {}


def solve_backtracking(solver: HashiSolver, time_limit: float = 30.0) -> Dict[Tuple, int]:
    """
    Optimized Backtracking solver.
    Uses MCV (Most Constrained Variable) ordering and index-based recursion
    to eliminate set-copying overhead.
    """
    import time
    start = time.time()

    # 1. Preprocessing: Build Adjacency and Pairs
    pairs_list = []
    seen = set()
    neighbors_map = {(i, j): [] for i, j, _ in solver.islands}

    # We need to map islands to their current degree for the heuristic
    island_degrees = {(i, j): 0 for i, j, _ in solver.islands}

    for i, j, _ in solver.islands:
        # Note: Ensure find_neighbors is robust
        neighs = solver.find_neighbors((i, j, 0))
        for ni, nj in neighs:
            island_degrees[(i, j)] += 1
            a, b = (i, j), (ni, nj)
            # Standardize key (lower coordinate first)
            key = (a, b) if a <= b else (b, a)
            if key not in seen:
                seen.add(key)
                pairs_list.append(key)
            neighbors_map[a].append((ni, nj))
            # Note: find_neighbors usually returns both directions,
            # but we ensure map is built correctly just in case
            if (i, j) not in neighbors_map.get((ni, nj), []):
                neighbors_map.setdefault((ni, nj), []).append((i, j))

    # 2. HEURISTIC SORTING (Crucial for performance)
    # Sort pairs based on the sum of degrees of the endpoints.
    # Logic: Islands with fewer neighbors are "harder" (more constrained).
    # We want to lock those bridges in early.
    pairs_list.sort(key=lambda p: island_degrees[p[0]] + island_degrees[p[1]])

    islands_req = {(i, j): req for i, j, req in solver.islands}
    island_list = list(islands_req.keys())

    # Pre-calculate count of pairs
    num_pairs = len(pairs_list)

    # --- Helpers ---

    def segments_overlap(a1, a2, b1, b2):
        (i1, j1), (i2, j2) = a1, a2
        (i3, j3), (i4, j4) = b1, b2
        # Horizontal overlap
        if i1 == i2 == i3 == i4:
            a_min, a_max = sorted((j1, j2))
            b_min, b_max = sorted((j3, j4))
            return not (a_max <= b_min or b_max <= a_min)
        # Vertical overlap
        if j1 == j2 == j3 == j4:
            a_min, a_max = sorted((i1, i2))
            b_min, b_max = sorted((i3, i4))
            return not (a_max <= b_min or b_max <= a_min)
        return False

    def valid_against_existing(a, b, new_count, existing):
        # 0 count bridges don't physically exist, so they can't cross/overlap
        if new_count <= 0: return True

        for (c1, c2), cnt in existing.items():
            if cnt <= 0: continue  # Ghost bridge

            # Crossing check
            try:
                if solver.bridges_cross((c1, c2), (a, b)): return False
            except Exception:
                pass  # Should not happen if data is clean

            # Collinear Overlap check (strict)
            if (c1[0] == c2[0] == a[0] == b[0]) or (c1[1] == c2[1] == a[1] == b[1]):
                if segments_overlap(c1, c2, a, b): return False
        return True

    def count_components_from_bridges(bridges_map):
        parent = {isl: isl for isl in island_list}

        def find(x):
            path = []
            while parent[x] != x:
                path.append(x)
                x = parent[x]
            for p in path: parent[p] = x
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra

        for (a, b), cnt in bridges_map.items():
            if cnt > 0: union(a, b)

        return len(set(find(x) for x in island_list))

    def check_remaining_capacity(idx, remaining):
        """
        Lookahead Pruning:
        Check if any island has more requirement left than its
        remaining available neighbors can possibly provide.
        """
        # We only need to check islands involved in the FUTURE pairs
        # But iterating all islands is costly.
        # A lightweight check: iterate neighbors of current pair?
        # For simplicity/safety, we do a full check but optimized.

        # We calculate max potential supply from UNASSIGNED edges
        # This is O(N_unassigned), which is better than O(Islands) if we are deep.
        # However, simple iteration over islands is often safer logic.

        supply = {k: 0 for k in remaining}

        # Iterate only remaining pairs in the list
        for k in range(idx, num_pairs):
            u, v = pairs_list[k]
            # The most an edge can give is 2, OR what the *other* node has left.
            # This is the "Tighter Pruning" improvement.
            supply[u] += min(2, remaining[v])
            supply[v] += min(2, remaining[u])

        for k, req in remaining.items():
            if req > supply[k]:
                return False
        return True

    # --- DFS State ---
    solution = {}
    found = False
    visited_nodes = 0

    def dfs(idx, bridges_map, remaining_map):
        nonlocal found, solution, visited_nodes
        if found: return

        # Check limits periodically (every 1000 nodes) to save overhead
        visited_nodes += 1
        if visited_nodes % 1000 == 0:
            if time_limit is not None and (time.time() - start) > time_limit:
                return

        # 1. Pruning: Negative requirements (Overfill)
        # (This is mostly handled by the loop guard, but good for safety)
        for k, v in remaining_map.items():
            if v < 0: return

        # 2. Base Case: All edges assigned
        if idx == num_pairs:
            # Check 1: Are all requirements exactly 0?
            if all(v == 0 for v in remaining_map.values()):
                # Check 2: Connectivity
                if count_components_from_bridges(bridges_map) == 1:
                    # Found! Copy solution
                    sol = {}
                    for (a, b), cnt in bridges_map.items():
                        if cnt > 0:
                            a_key, b_key = (a, b) if a <= b else (b, a)
                            sol[(a_key, b_key, cnt)] = 1
                    solution = sol
                    found = True
            return

        # 3. Lookahead Pruning
        if not check_remaining_capacity(idx, remaining_map):
            return

        # 4. Recursive Step
        pair = pairs_list[idx]
        u, v = pair

        # HEURISTIC VALUE ORDERING:
        # Usually, trying larger bridges first (2) helps satisfy constraints faster,
        # but 0 is also important to cut off branches.
        # 2 -> 1 -> 0 is standard.
        for val in (2, 1, 0):
            # Constraint: Don't exceed island capacity
            if remaining_map[u] < val or remaining_map[v] < val:
                if val != 0: continue  # Must pick 0 if we can't afford 1 or 2

            # Constraint: Geometry (Crossings)
            # Only need to check if we are placing actual bridges (>0)
            if val > 0:
                if not valid_against_existing(u, v, val, bridges_map):
                    continue

            # Apply Move
            bridges_map[pair] = val
            remaining_map[u] -= val
            remaining_map[v] -= val

            dfs(idx + 1, bridges_map, remaining_map)

            # Backtrack
            if found: return
            remaining_map[u] += val
            remaining_map[v] += val
            del bridges_map[pair]

    # Initialize
    init_bridges = {}
    init_remaining = {k: v for k, v in islands_req.items()}

    try:
        dfs(0, init_bridges, init_remaining)
    except RecursionError:
        print("Recursion limit hit. Increase sys.setrecursionlimit if needed.")

    elapsed = time.time() - start
    if found:
        print(f"Backtracking: found solution in {elapsed:.3f}s (visited {visited_nodes} nodes)")
        return solution
    else:
        print(f"Backtracking: no solution found after {elapsed:.3f}s (visited {visited_nodes} nodes)")
        return {}


###############################################################################
# HELPERS & MAIN
###############################################################################

def read_input_file(filename: str) -> np.ndarray:
    """Read input file and convert to grid"""
    input_path = os.path.join('Inputs', filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Parse the grid
    grid_data = []
    for line in lines:
        # Remove whitespace and split by commas or spaces
        line = line.strip()
        if line:
            # Try comma-separated first
            if ',' in line:
                row = [int(x.strip()) for x in line.split(',')]
            else:
                # Space-separated
                row = [int(x) for x in line.split()]
            grid_data.append(row)

    return np.array(grid_data)


def write_output_file(filename: str, output: List[List[str]]):
    """Write output to file with each item quoted."""
    output_dir = 'Outputs'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w') as f:
        for row in output:
            quoted_row = ['"' + cell + '"' for cell in row]  # add quotes
            f.write('[' + ', '.join(quoted_row) + ']\n')

    print(f"Output written to: {output_path}")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Hashiwokakero Solver")
    print("=" * 60)

    filename = input("Enter input filename (e.g., input-01.txt): ").strip()

    try:
        # Read input
        grid = read_input_file(filename)
        print(f"\nLoaded grid from {filename}:")
        print(grid)

        # Create solver
        solver = HashiSolver(grid)

        # Choose solving method
        print("\nChoose solving method:")
        print("1. A* Search")
        print("2. Backtracking (DFS + heuristics)")
        print("3. Brute-force")
        print("4. PySAT")

        choice = input("Enter choice (1-4): ").strip()

        ###############################
        # Run the chosen algorithm
        ###############################
        start = time.time()
        solution = {}

        if choice == "1":
            print("\nSolving with A* search...")
            solution = solve_with_astar(solver)

        elif choice == "2":
            print("\nSolving with Backtracking (DFS + heuristics)...")
            solution = solve_backtracking(solver)

        elif choice == "3":
            print("\nSolving with Brute-force...")
            solution = solve_bruteforce(solver)

        elif choice == "4":
            print("\nSolving with PySAT...")
            solution = solve_with_pysat(solver)

        else:
            print("\nInvalid choice.")
            exit()

        end = time.time()
        elapsed_ms = (end - start) * 1000

        ###############################
        # Output
        ###############################
        if solution:
            print(f"\n✓ Solution found! (time = {elapsed_ms:.2f} ms)")
            output = solver.format_output(solution)

            print("\nSolution grid:")
            for row in output:
                print(row)

            output_filename = filename.replace('input', 'output').replace('.txt', '.txt')
            write_output_file(output_filename, output)

        else:
            print(f"\n✗ No solution found (time = {elapsed_ms:.2f} ms)")

    except FileNotFoundError as e:
        print(f"\nError: {e}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()