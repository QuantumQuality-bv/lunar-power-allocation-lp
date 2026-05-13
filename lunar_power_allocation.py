"""Run the Math 464 lunar power-allocation study.

The script verifies two custom simplex implementations, builds three lunar
base power-allocation datasets, compares tableau and revised simplex results
with a PuLP/CBC baseline, and writes the project figures to the figures/
directory.
"""

# Math 464 final project script.
# Part 1 checks the custom simplex code on small general-form LPs.
# Part 2 builds the lunar power-allocation LP datasets.
# Part 3 solves the datasets with tableau simplex, revised simplex, and PuLP/CBC.

import time
from pathlib import Path

import numpy as np
import pulp
import matplotlib.pyplot as plt


# Generated figures are written here when the script is run.
FIGURES_DIR = Path("figures")

# Numerical tolerance used to treat very small values as zero.
TOL = 1e-9

# Two operating modes: an active crewed mode and a lower-power dormant mode.
MODES = ["crewed", "dormant"]
DATASET_ORDER = ["small", "medium", "large"]

MODE_HOURS = {
    "crewed": 720.0,
    "dormant": 8040.0,
}

TOTAL_AVAILABLE_POWER = {
    "crewed": 50.0,
    "dormant": 40.0,
}

FIXED_HABITAT_KEEP_ALIVE = {
    "crewed": 2.0,
    "dormant": 2.0,
}

# The fixed keep-alive habitat load is removed before optimizing flexible loads.
FLEXIBLE_AVAILABLE_POWER = {
    mode: TOTAL_AVAILABLE_POWER[mode] - FIXED_HABITAT_KEEP_ALIVE[mode]
    for mode in MODES
}

# Each dataset adds more flexible loads. The larger cases keep the same LP structure but increase the number of variables and constraints.
DATASETS = {
    "small": {
        "report_load_count": 4,
        "flexible_loads": ["h1", "r1", "s"],
        "demand": {
            "h1": {"crewed": 18.0, "dormant": 0.0},
            "r1": {"crewed": 0.5, "dormant": 0.0},
            "s": {"crewed": 0.2, "dormant": 0.0},
        },
        "efficiency": {
            "h1": 1.0,
            "r1": 1.0,
            "s": 1.0,
        },
        "penalty": {
            "h1": 100.0,
            "r1": 25.0,
            "s": 10.0,
        },
    },
    "medium": {
        "report_load_count": 5,
        "flexible_loads": ["h1", "r1", "r2", "s"],
        "demand": {
            "h1": {"crewed": 18.0, "dormant": 0.0},
            "r1": {"crewed": 0.5, "dormant": 0.0},
            "r2": {"crewed": 0.5, "dormant": 0.0},
            "s": {"crewed": 0.2, "dormant": 0.0},
        },
        "efficiency": {
            "h1": 1.0,
            "r1": 1.0,
            "r2": 0.95,
            "s": 1.0,
        },
        "penalty": {
            "h1": 100.0,
            "r1": 25.0,
            "r2": 25.0,
            "s": 10.0,
        },
    },
    "large": {
        "report_load_count": 7,
        "flexible_loads": ["h1", "r1", "r2", "s", "ir", "ic"],
        "demand": {
            "h1": {"crewed": 18.0, "dormant": 0.0},
            "r1": {"crewed": 0.5, "dormant": 0.0},
            "r2": {"crewed": 0.5, "dormant": 0.0},
            "s": {"crewed": 0.2, "dormant": 0.0},
            "ir": {"crewed": 46.0, "dormant": 8.0},
            "ic": {"crewed": 22.0, "dormant": 4.0},
        },
        "efficiency": {
            "h1": 1.0,
            "r1": 1.0,
            "r2": 0.95,
            "s": 1.0,
            "ir": 0.92,
            "ic": 0.90,
        },
        "penalty": {
            "h1": 100.0,
            "r1": 25.0,
            "r2": 25.0,
            "s": 10.0,
            "ir": 8.0,
            "ic": 8.0,
        },
    },
}


# ~~~~~~~~~~~~~~
# basic utilities

def print_separator(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def cleaned_vector(x: np.ndarray) -> np.ndarray:
    # Round tiny numerical noise to zero so printed solutions are easier to read.
    y = np.array(x, dtype=float, copy=True)
    y[np.abs(y) < TOL] = 0.0
    return y


def value_by_name(var_names, x):
    # Pair variable names with their numerical values for readable output.
    return {name: float(x[i]) for i, name in enumerate(var_names)}


def choose_entering_bland_from_reduced_costs(reduced_costs):
    # For min, a negative reduced cost can improve the objective.
    # Bland-style tie-breaking chooses the lowest-index improving variable.
    candidates = [j for j, value in enumerate(reduced_costs) if value < -TOL]
    return min(candidates) if candidates else None


def choose_leaving_bland(x_B, direction, basis):
    # Ratio test: only positive direction entries can limit the step length.
    # The basis index is included in the tuple to break ties consistently.
    candidates = []
    for row, value in enumerate(direction):
        if value > TOL:
            ratio = x_B[row] / value
            candidates.append((ratio, basis[row], row))
    return min(candidates)[2] if candidates else None


# ~~~~~~~~~~~~~~
# part 1: general LP input, preprocessing, and two-phase setup

def make_general_lp(name, c, A, b, senses, var_names):
    # Store a general-form minimization LP in one dictionary.
    return {
        "name": name,
        "sense": "min",
        "c": np.array(c, dtype=float),
        "A": np.array(A, dtype=float),
        "b": np.array(b, dtype=float),
        "senses": list(senses),
        "var_names": list(var_names),
    }


def normalize_general_lp(lp):
    # Put the LP into a cleaner form first. If a row has a negative right-hand side, flip the row and reverse the inequality direction.
    A = np.array(lp["A"], dtype=float, copy=True)
    b = np.array(lp["b"], dtype=float, copy=True)
    senses = list(lp["senses"])

    for i in range(len(b)):
        if b[i] < -TOL:
            A[i, :] *= -1.0
            b[i] *= -1.0
            if senses[i] == "<=":
                senses[i] = ">="
            elif senses[i] == ">=":
                senses[i] = "<="

    return {
        "name": lp["name"],
        "sense": lp["sense"],
        "c": np.array(lp["c"], dtype=float, copy=True),
        "A": A,
        "b": b,
        "senses": senses,
        "var_names": list(lp["var_names"]),
    }


def convert_general_lp_to_standard(lp):
    # Build the standard-form version used by simplex.
    # Slack variables handle <= rows; surplus plus artificial variables handle >= rows.
    # Equality rows get artificial variables when they are needed for the Phase I basis.
    lp = normalize_general_lp(lp)

    A = np.array(lp["A"], dtype=float, copy=True)
    b = np.array(lp["b"], dtype=float, copy=True)
    c = np.array(lp["c"], dtype=float, copy=True)
    senses = list(lp["senses"])
    var_names = list(lp["var_names"])

    m, n = A.shape
    basis = [None] * m
    artificial_cols = []

    A_std = A.copy()
    phase2_cost = list(c)

    for row, sense in enumerate(senses):
        if sense == "<=":
            # A <= row gets a positive slack variable, which can enter the starting basis.
            col = np.zeros(m, dtype=float)
            col[row] = 1.0
            A_std = np.column_stack((A_std, col))
            var_names.append(f"slack_{row + 1}")
            phase2_cost.append(0.0)
            basis[row] = A_std.shape[1] - 1

        elif sense == ">=":
            # A >= row needs a surplus variable to make equality form.
            # The surplus column alone is not a basic column, so add an artificial variable.
            col = np.zeros(m, dtype=float)
            col[row] = -1.0
            A_std = np.column_stack((A_std, col))
            var_names.append(f"surplus_{row + 1}")
            phase2_cost.append(0.0)

            col = np.zeros(m, dtype=float)
            col[row] = 1.0
            A_std = np.column_stack((A_std, col))
            var_names.append(f"artificial_{row + 1}")
            phase2_cost.append(0.0)
            basis[row] = A_std.shape[1] - 1
            artificial_cols.append(A_std.shape[1] - 1)

        elif sense == "=":
            # Equality rows may not have an obvious starting basic variable, so an artificial variable is used for Phase I.
            col = np.zeros(m, dtype=float)
            col[row] = 1.0
            A_std = np.column_stack((A_std, col))
            var_names.append(f"artificial_{row + 1}")
            phase2_cost.append(0.0)
            basis[row] = A_std.shape[1] - 1
            artificial_cols.append(A_std.shape[1] - 1)

        else:
            raise ValueError(f"Unsupported constraint sense: {sense}")

    # Phase I minimizes the sum of artificial variables.
    phase1_cost = np.zeros(A_std.shape[1], dtype=float)
    for j in artificial_cols:
        phase1_cost[j] = 1.0

    return {
        "name": lp["name"],
        "A": A_std,
        "b": b,
        "c_phase2": np.array(phase2_cost, dtype=float),
        "c_phase1": phase1_cost,
        "basis": list(basis),
        "var_names": var_names,
        "artificial_cols": list(artificial_cols),
        "original_var_count": n,
    }


# ~~~~~~~~~~~~~~
# tableau simplex

def build_canonical_tableau(A, b, c, basis):
    # Build the tableau associated with the current basis.
    # The bottom row stores reduced costs for the minimization optimality test.
    B = A[:, basis]
    B_inv = np.linalg.inv(B)
    body = B_inv @ A
    rhs = B_inv @ b
    c_B = c[basis]
    reduced_costs = c - c_B @ body

    tableau = np.zeros((A.shape[0] + 1, A.shape[1] + 1), dtype=float)
    tableau[:-1, :-1] = body
    tableau[:-1, -1] = rhs
    tableau[-1, :-1] = reduced_costs
    tableau[np.abs(tableau) < TOL] = 0.0
    return tableau


def pivot_tableau(tableau, pivot_row, pivot_col):
    # Standard row operation pivot: scale the pivot row, then clear the pivot column.
    pivot = tableau[pivot_row, pivot_col]
    tableau[pivot_row, :] = tableau[pivot_row, :] / pivot

    for row in range(tableau.shape[0]):
        if row != pivot_row:
            tableau[row, :] = tableau[row, :] - tableau[row, pivot_col] * tableau[pivot_row, :]

    tableau[np.abs(tableau) < TOL] = 0.0


def tableau_simplex_standard_min(A, b, c, basis, max_iter=500):
    # Tableau simplex for a standard-form minimization LP with a given feasible basis.
    basis = list(basis)
    tableau = build_canonical_tableau(A, b, c, basis)

    for iteration in range(max_iter):
        entering = choose_entering_bland_from_reduced_costs(tableau[-1, :-1])

        if entering is None:
            # No negative reduced costs remain, so the current BFS is optimal.
            x = np.zeros(A.shape[1], dtype=float)
            for row, basic_index in enumerate(basis):
                x[basic_index] = tableau[row, -1]
            x = cleaned_vector(x)
            return {
                "status": "optimal",
                "x": x,
                "value": float(c @ x),
                "basis": basis.copy(),
                "iterations": iteration,
            }

        leaving_row = choose_leaving_bland(tableau[:-1, -1], tableau[:-1, entering], basis)
        if leaving_row is None:
            # If there is no leaving row, the objective can improve without bound.
            return {
                "status": "unbounded",
                "basis": basis.copy(),
                "iterations": iteration,
            }

        pivot_tableau(tableau, leaving_row, entering)
        basis[leaving_row] = entering

    return {
        "status": "iteration_limit",
        "basis": basis.copy(),
        "iterations": max_iter,
    }


# ~~~~~~~~~~~~~~
# revised simplex

def revised_simplex_standard_min(A, b, c, basis, max_iter=500):
    # Revised simplex solves the same standard-form problem without storing or updating the full tableau.
    basis = list(basis)

    for iteration in range(max_iter):
        B = A[:, basis]
        x_B = np.linalg.solve(B, b)
        c_B = c[basis]
        lamb = np.linalg.solve(B.T, c_B)

        # Reduced costs are only needed for nonbasic variables.
        reduced_costs = np.full(A.shape[1], np.inf, dtype=float)
        for j in range(A.shape[1]):
            if j not in basis:
                reduced_costs[j] = c[j] - A[:, j] @ lamb

        entering = choose_entering_bland_from_reduced_costs(reduced_costs)

        if entering is None:
            # All nonbasic reduced costs are nonnegative, so the current BFS is optimal.
            x = np.zeros(A.shape[1], dtype=float)
            for row, basic_index in enumerate(basis):
                x[basic_index] = x_B[row]
            x = cleaned_vector(x)
            return {
                "status": "optimal",
                "x": x,
                "value": float(c @ x),
                "basis": basis.copy(),
                "iterations": iteration,
            }

        # The direction tells how basic variables change when the entering variable increases.
        direction = np.linalg.solve(B, A[:, entering])
        leaving_row = choose_leaving_bland(x_B, direction, basis)
        if leaving_row is None:
            return {
                "status": "unbounded",
                "basis": basis.copy(),
                "iterations": iteration,
            }

        basis[leaving_row] = entering

    return {
        "status": "iteration_limit",
        "basis": basis.copy(),
        "iterations": max_iter,
    }


# ~~~~~~~~~~~~~~
# two-phase wrapper for part 1

def cleanup_phase_one_basis(A, b, c_phase2, basis, var_names, artificial_cols):
    # After Phase I succeeds, remove artificial variables before solving Phase II.
    # If an artificial variable is basic at zero, try to pivot it out. If no pivot
    # is possible, the row is redundant and can be removed.
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    c_phase2 = np.array(c_phase2, dtype=float, copy=True)
    basis = list(basis)
    var_names = list(var_names)
    artificial_set = set(artificial_cols)

    while True:
        B = A[:, basis]
        B_inv = np.linalg.inv(B)
        body = B_inv @ A
        rhs = B_inv @ b

        changed = False
        for row, basic in enumerate(basis):
            if basic in artificial_set:
                if abs(rhs[row]) > 1e-7:
                    raise RuntimeError("Phase I cleanup failed: basic artificial variable is positive.")

                candidates = []
                for j in range(A.shape[1]):
                    if j not in artificial_set and j not in basis and abs(body[row, j]) > TOL:
                        candidates.append(j)

                if candidates:
                    basis[row] = min(candidates)
                    changed = True
                    break

                keep_rows = [r for r in range(A.shape[0]) if r != row]
                A = A[keep_rows, :]
                b = b[keep_rows]
                basis.pop(row)
                changed = True
                break

        if not changed:
            break

    # Once artificial variables are out of the basis, delete their columns.
    keep_cols = [j for j in range(A.shape[1]) if j not in artificial_set]
    old_to_new = {old: new for new, old in enumerate(keep_cols)}

    A = A[:, keep_cols]
    c_phase2 = c_phase2[keep_cols]
    var_names = [var_names[j] for j in keep_cols]
    basis = [old_to_new[j] for j in basis]

    return A, b, c_phase2, basis, var_names


def solve_general_lp_with_simplex(lp, method_name):
    # Solve a general-form LP by converting it to standard form, running Phase I if needed, and then solving the original objective in Phase II.
    standard = convert_general_lp_to_standard(lp)

    A = standard["A"]
    b = standard["b"]
    c_phase1 = standard["c_phase1"]
    c_phase2 = standard["c_phase2"]
    basis = standard["basis"]
    var_names = standard["var_names"]
    artificial_cols = standard["artificial_cols"]
    original_var_count = standard["original_var_count"]

    if method_name == "tableau":
        simplex_method = tableau_simplex_standard_min
    elif method_name == "revised":
        simplex_method = revised_simplex_standard_min
    else:
        raise ValueError("method_name must be 'tableau' or 'revised'.")

    phase1_iterations = 0
    phase2_iterations = 0

    if artificial_cols:
        phase1 = simplex_method(A, b, c_phase1, basis)
        phase1_iterations = int(phase1["iterations"])

        if phase1["status"] != "optimal":
            return {
                "status": phase1["status"],
                "phase1_iterations": phase1_iterations,
                "phase2_iterations": phase2_iterations,
            }

        # Phase I checks whether artificial variables can be driven to zero. If not, the original LP has no feasible solution.
        if phase1["value"] > 1e-7:
            return {
                "status": "infeasible",
                "phase1_iterations": phase1_iterations,
                "phase2_iterations": phase2_iterations,
            }

        A, b, c_phase2, basis, var_names = cleanup_phase_one_basis(
            A, b, c_phase2, phase1["basis"], var_names, artificial_cols
        )

    phase2 = simplex_method(A, b, c_phase2, basis)
    phase2_iterations = int(phase2["iterations"])

    if phase2["status"] != "optimal":
        return {
            "status": phase2["status"],
            "phase1_iterations": phase1_iterations,
            "phase2_iterations": phase2_iterations,
        }

    x_std = phase2["x"]
    x_original = x_std[:original_var_count]

    return {
        "status": "optimal",
        "x": cleaned_vector(x_original),
        "objective": float(np.array(lp["c"], dtype=float) @ x_original),
        "var_names": list(lp["var_names"]),
        "standard_form_variable_count": A.shape[1],
        "standard_form_constraint_count": A.shape[0],
        "phase1_iterations": phase1_iterations,
        "phase2_iterations": phase2_iterations,
        "total_iterations": phase1_iterations + phase2_iterations,
    }


# ~~~~~~~~~~~~~~
# part 1: demo problems

def build_part1_example_optimal():
    # Feasible lower-bound example. This mainly checks >= rows, surplus variables, artificial variables, and successful Phase I cleanup.
    return make_general_lp(
        name="FeasibleLowerBoundExample",
        c=[3.0, 4.0],
        A=[
            [2.0, 3.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        b=[13.0, 2.0, 5.0],
        senses=[">=", ">=", ">="],
        var_names=["x1", "x2"],
    )


def build_part1_example_mixed_sense():
    # This test checks a feasible LP with <=, =, and >= constraints in one model.
    return make_general_lp(
        name="MixedSenseFeasibleExample",
        c=[1.0, 2.0],
        A=[
            [1.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.0],
        ],
        b=[6.0, 1.0, 3.0],
        senses=["<=", "=", ">="],
        var_names=["x1", "x2"],
    )


def build_part1_example_infeasible():
    # Infeasible example. Phase I should detect that x1 >= 2 and x1 <= 1 cannot both be satisfied.
    return make_general_lp(
        name="InfeasibleExample",
        c=[1.0],
        A=[
            [1.0],
            [1.0],
        ],
        b=[2.0, 1.0],
        senses=[">=", "<="],
        var_names=["x1"],
    )


def run_part1_checks():
    print_separator("SECTION 1: PART 1 GENERAL SIMPLEX IMPLEMENTATION CHECKS")

    examples = [
        ("Feasible lower-bound LP", build_part1_example_optimal()),
        ("Mixed-sense feasible LP", build_part1_example_mixed_sense()),
        ("Simple infeasible LP", build_part1_example_infeasible()),
    ]

    for label, lp in examples:
        print(label)
        for method_name in ["tableau", "revised"]:
            start_time = time.perf_counter()
            result = solve_general_lp_with_simplex(lp, method_name)
            end_time = time.perf_counter()

            print(f"  {method_name.capitalize():8s} | status = {result['status']}", end="")

            if result["status"] == "optimal":
                solution_map = value_by_name(result["var_names"], result["x"])
                print(
                    f" | objective = {result['objective']:.6f}"
                    f" | x = {solution_map}"
                    f" | std vars = {result['standard_form_variable_count']}"
                    f" | std cons = {result['standard_form_constraint_count']}"
                    f" | phase1 = {result['phase1_iterations']}"
                    f" | phase2 = {result['phase2_iterations']}"
                    f" | runtime = {end_time - start_time:.6f}s"
                )
            else:
                print(
                    f" | phase1 = {result.get('phase1_iterations', 0)}"
                    f" | phase2 = {result.get('phase2_iterations', 0)}"
                    f" | runtime = {end_time - start_time:.6f}s"
                )
        print()


# ~~~~~~~~~~~~~~
# part 2: lunar model in general form

def lunar_variable_order(loads):
    # General-form variables are grouped as dispatch, delivered power, and shortage.
    names = []
    for prefix in ["Dispatch", "Delivered", "Shortage"]:
        for load in loads:
            for mode in MODES:
                names.append(f"{prefix}_{load}_{mode}")
    return names


def lunar_index_map(var_names):
    return {name: i for i, name in enumerate(var_names)}


def build_lunar_general_lp(dataset_name):
    # Build the lunar LP in the natural model form used by PuLP/CBC: capacity inequalities, delivery equalities, and demand-balance equalities.
    dataset = DATASETS[dataset_name]
    loads = dataset["flexible_loads"]
    var_names = lunar_variable_order(loads)
    idx = lunar_index_map(var_names)

    A_rows = []
    b_rows = []
    senses = []

    # Source-capacity constraints for crewed and dormant modes.
    for mode in MODES:
        row = np.zeros(len(var_names), dtype=float)
        for load in loads:
            row[idx[f"Dispatch_{load}_{mode}"]] = 1.0
        A_rows.append(row)
        b_rows.append(FLEXIBLE_AVAILABLE_POWER[mode])
        senses.append("<=")

    # Delivery equations: delivered power equals efficiency times dispatched power.
    for load in loads:
        for mode in MODES:
            row = np.zeros(len(var_names), dtype=float)
            row[idx[f"Delivered_{load}_{mode}"]] = 1.0
            row[idx[f"Dispatch_{load}_{mode}"]] = -dataset["efficiency"][load]
            A_rows.append(row)
            b_rows.append(0.0)
            senses.append("=")

    # Demand balance: delivered power plus shortage equals demand.
    for load in loads:
        for mode in MODES:
            row = np.zeros(len(var_names), dtype=float)
            row[idx[f"Delivered_{load}_{mode}"]] = 1.0
            row[idx[f"Shortage_{load}_{mode}"]] = 1.0
            A_rows.append(row)
            b_rows.append(dataset["demand"][load][mode])
            senses.append("=")

    # Objective penalizes shortage, weighted by operating-mode duration and load priority.
    c = np.zeros(len(var_names), dtype=float)
    for load in loads:
        for mode in MODES:
            c[idx[f"Shortage_{load}_{mode}"]] = MODE_HOURS[mode] * dataset["penalty"][load]

    lp = make_general_lp(
        name=f"LunarPower_{dataset_name}",
        c=c,
        A=np.array(A_rows, dtype=float),
        b=np.array(b_rows, dtype=float),
        senses=senses,
        var_names=var_names,
    )

    return lp, dataset


def build_lunar_benchmark_standard_form(dataset_name):
    # Build the standard-form lunar matrix used by the custom simplex methods.
    dataset = DATASETS[dataset_name]
    loads = dataset["flexible_loads"]
    n = len(loads)

    variable_names = []
    for prefix in ["Dispatch_c", "Dispatch_d", "Delivered_c", "Delivered_d", "Shortage_c", "Shortage_d"]:
        for load in loads:
            variable_names.append(f"{prefix}_{load}")
    variable_names += ["Slack_capacity_c", "Slack_capacity_d"]

    index = {}
    p = 0
    for prefix in ["Dispatch_c", "Dispatch_d", "Delivered_c", "Delivered_d", "Shortage_c", "Shortage_d"]:
        for load in loads:
            index[(prefix, load)] = p
            p += 1
    index["Slack_capacity_c"] = p
    p += 1
    index["Slack_capacity_d"] = p

    num_rows = 2 + 4 * n
    num_cols = len(variable_names)

    A = np.zeros((num_rows, num_cols), dtype=float)
    b = np.zeros(num_rows, dtype=float)
    c = np.zeros(num_cols, dtype=float)

    # Capacity rows use slack variables to become equalities.
    for load in loads:
        A[0, index[("Dispatch_c", load)]] = 1.0
        A[1, index[("Dispatch_d", load)]] = 1.0
    A[0, index["Slack_capacity_c"]] = 1.0
    A[1, index["Slack_capacity_d"]] = 1.0
    b[0] = FLEXIBLE_AVAILABLE_POWER["crewed"]
    b[1] = FLEXIBLE_AVAILABLE_POWER["dormant"]

    # Crewed delivery rows.
    for j, load in enumerate(loads):
        row = 2 + j
        A[row, index[("Delivered_c", load)]] = 1.0
        A[row, index[("Dispatch_c", load)]] = -dataset["efficiency"][load]

    # Dormant delivery rows.
    for j, load in enumerate(loads):
        row = 2 + n + j
        A[row, index[("Delivered_d", load)]] = 1.0
        A[row, index[("Dispatch_d", load)]] = -dataset["efficiency"][load]

    # Crewed demand-balance rows and shortage costs.
    for j, load in enumerate(loads):
        row = 2 + 2 * n + j
        A[row, index[("Delivered_c", load)]] = 1.0
        A[row, index[("Shortage_c", load)]] = 1.0
        b[row] = dataset["demand"][load]["crewed"]
        c[index[("Shortage_c", load)]] = MODE_HOURS["crewed"] * dataset["penalty"][load]

    # Dormant demand-balance rows and shortage costs.
    for j, load in enumerate(loads):
        row = 2 + 3 * n + j
        A[row, index[("Delivered_d", load)]] = 1.0
        A[row, index[("Shortage_d", load)]] = 1.0
        b[row] = dataset["demand"][load]["dormant"]
        c[index[("Shortage_d", load)]] = MODE_HOURS["dormant"] * dataset["penalty"][load]

    # The lunar benchmark starts from an obvious feasible basis: dispatch and delivered variables start at zero, shortages meet demand, and capacity slacks satisfy the source limits.
    initial_basis = (
        [index["Slack_capacity_c"], index["Slack_capacity_d"]]
        + [index[("Delivered_c", load)] for load in loads]
        + [index[("Delivered_d", load)] for load in loads]
        + [index[("Shortage_c", load)] for load in loads]
        + [index[("Shortage_d", load)] for load in loads]
    )

    return {
        "A": A,
        "b": b,
        "c": c,
        "var_names": variable_names,
        "basis": list(initial_basis),
        "dataset": dataset,
    }


def run_part2_summary():
    print_separator("SECTION 2: PART 2 LUNAR MODEL AND DATASET SUMMARY")
    print("Locked operating modes:", MODES)
    print("Total source-side power:", TOTAL_AVAILABLE_POWER)
    print("Fixed habitat keep-alive:", FIXED_HABITAT_KEEP_ALIVE)
    print("Flexible source-side power:", FLEXIBLE_AVAILABLE_POWER)
    print()

    for dataset_name in DATASET_ORDER:
        general_lp, dataset = build_lunar_general_lp(dataset_name)
        benchmark = build_lunar_benchmark_standard_form(dataset_name)
        print(f"Dataset: {dataset_name}")
        print(f"  Report load count: {dataset['report_load_count']}")
        print(f"  Flexible loads: {dataset['flexible_loads']}")
        print(f"  General LP variables: {len(general_lp['var_names'])}")
        print(f"  General LP constraints: {len(general_lp['senses'])}")
        print(f"  Benchmark standard-form variables: {benchmark['A'].shape[1]}")
        print(f"  Benchmark standard-form constraints: {benchmark['A'].shape[0]}")
        print()


# ~~~~~~~~~~~~~~
# part 3: solve, extract, print, summarize

def solve_general_lp_with_pulp(lp):
    # PuLP/CBC is used only as a standard-solver baseline.
    # The custom tableau and revised simplex methods do not call this function.
    problem = pulp.LpProblem(lp["name"], pulp.LpMinimize)
    variables = {
        name: pulp.LpVariable(name, lowBound=0, cat=pulp.LpContinuous)
        for name in lp["var_names"]
    }

    problem += pulp.lpSum(
        lp["c"][j] * variables[lp["var_names"][j]]
        for j in range(len(lp["var_names"]))
    ), "Objective"

    for i, sense in enumerate(lp["senses"]):
        expr = pulp.lpSum(
            lp["A"][i, j] * variables[lp["var_names"][j]]
            for j in range(len(lp["var_names"]))
        )
        rhs = lp["b"][i]

        if sense == "<=":
            problem += (expr <= rhs, f"row_{i + 1}")
        elif sense == ">=":
            problem += (expr >= rhs, f"row_{i + 1}")
        elif sense == "=":
            problem += (expr == rhs, f"row_{i + 1}")
        else:
            raise ValueError(f"Unsupported constraint sense: {sense}")

    start_time = time.perf_counter()
    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    end_time = time.perf_counter()

    x = np.array([variables[name].varValue for name in lp["var_names"]], dtype=float)
    x = cleaned_vector(x)

    return {
        "status": pulp.LpStatus[problem.status].lower(),
        "x": x,
        "objective": float(pulp.value(problem.objective)),
        "runtime_seconds": end_time - start_time,
        "variable_count": len(problem.variables()),
        "constraint_count": len(problem.constraints),
        "iterations": None,
    }


def unpack_general_lunar_solution(dataset_name, x_general):
    # Convert the flat PuLP solution vector back into dispatch, delivered, shortage, and capacity-slack dictionaries.
    general_lp, dataset = build_lunar_general_lp(dataset_name)
    value_map = value_by_name(general_lp["var_names"], x_general)

    results = {
        "dataset": dataset_name,
        "report_load_count": dataset["report_load_count"],
        "dispatch": {},
        "delivered": {},
        "shortage": {},
        "capacity_slack": {},
    }

    for load in dataset["flexible_loads"]:
        results["dispatch"][load] = {}
        results["delivered"][load] = {}
        results["shortage"][load] = {}
        for mode in MODES:
            results["dispatch"][load][mode] = value_map[f"Dispatch_{load}_{mode}"]
            results["delivered"][load][mode] = value_map[f"Delivered_{load}_{mode}"]
            results["shortage"][load][mode] = value_map[f"Shortage_{load}_{mode}"]

    for mode in MODES:
        used = sum(results["dispatch"][load][mode] for load in dataset["flexible_loads"])
        results["capacity_slack"][mode] = FLEXIBLE_AVAILABLE_POWER[mode] - used

    return results


def unpack_benchmark_lunar_solution(dataset_name, x_standard):
    # Convert the standard-form simplex solution back into the same reporting format used for the PuLP solution.
    benchmark = build_lunar_benchmark_standard_form(dataset_name)
    dataset = benchmark["dataset"]
    value_map = value_by_name(benchmark["var_names"], x_standard)

    results = {
        "dataset": dataset_name,
        "report_load_count": dataset["report_load_count"],
        "dispatch": {},
        "delivered": {},
        "shortage": {},
        "capacity_slack": {
            "crewed": value_map["Slack_capacity_c"],
            "dormant": value_map["Slack_capacity_d"],
        },
    }

    loads = dataset["flexible_loads"]
    for load in loads:
        results["dispatch"][load] = {
            "crewed": value_map[f"Dispatch_c_{load}"],
            "dormant": value_map[f"Dispatch_d_{load}"],
        }
        results["delivered"][load] = {
            "crewed": value_map[f"Delivered_c_{load}"],
            "dormant": value_map[f"Delivered_d_{load}"],
        }
        results["shortage"][load] = {
            "crewed": value_map[f"Shortage_c_{load}"],
            "dormant": value_map[f"Shortage_d_{load}"],
        }

    return results


def build_method_result_record(dataset_name, method_name, solve_result, unpacked_results):
    # Merge solver metadata with the unpacked lunar allocation values.
    record = {
        "dataset": dataset_name,
        "method_name": method_name,
        "status": solve_result["status"],
        "objective": solve_result["objective"],
        "runtime_seconds": solve_result["runtime_seconds"],
        "variable_count": solve_result["variable_count"],
        "constraint_count": solve_result["constraint_count"],
        "iterations": solve_result["iterations"],
    }
    record.update(unpacked_results)
    return record


def print_lunar_dataset_results(result):
    print("-" * 78)
    print(f"Dataset: {result['dataset']}")
    print(f"Method: {result['method_name']}")
    print(f"Status: {result['status']}")
    print(f"Objective value: {result['objective']}")
    print(f"Runtime (seconds): {result['runtime_seconds']}")
    if result["iterations"] is not None:
        print(f"Iterations: {result['iterations']}")
    print(f"Variable count: {result['variable_count']}")
    print(f"Constraint count: {result['constraint_count']}")
    print(f"Report load count: {result['report_load_count']}")
    print(f"Capacity slack: {result['capacity_slack']}")
    print()

    print("Dispatch values")
    for load_name, mode_dict in result["dispatch"].items():
        print(load_name, mode_dict)
    print()

    print("Delivered values")
    for load_name, mode_dict in result["delivered"].items():
        print(load_name, mode_dict)
    print()

    print("Shortage values")
    for load_name, mode_dict in result["shortage"].items():
        print(load_name, mode_dict)
    print()


def print_method_summary(label, results):
    # Print the compact summary table used to check model size, runtime, objective value, iteration count, and status.
    header = "dataset | variables | constraints | runtime_seconds | objective"
    if any(item["iterations"] is not None for item in results):
        header += " | iterations"
    header += " | status"

    print(label)
    print(header)

    for item in results:
        parts = [
            str(item["dataset"]),
            str(item["variable_count"]),
            str(item["constraint_count"]),
            str(item["runtime_seconds"]),
            str(item["objective"]),
        ]
        if item["iterations"] is not None:
            parts.append(str(item["iterations"]))
        parts.append(str(item["status"]))
        print(" | ".join(parts))
    print()


def solve_lunar_dataset_with_method(dataset_name, method_key):
    # Route one dataset through either PuLP/CBC or one of the custom simplex methods.
    if method_key == "solver":
        lp, _ = build_lunar_general_lp(dataset_name)
        solve_result = solve_general_lp_with_pulp(lp)
        unpacked = unpack_general_lunar_solution(dataset_name, solve_result["x"])
        return build_method_result_record(dataset_name, "PuLP/CBC solver", solve_result, unpacked)

    benchmark = build_lunar_benchmark_standard_form(dataset_name)
    A = benchmark["A"]
    b = benchmark["b"]
    c = benchmark["c"]
    basis = benchmark["basis"]

    if method_key == "tableau":
        solver = tableau_simplex_standard_min
        method_name = "Tableau simplex"
    elif method_key == "revised":
        solver = revised_simplex_standard_min
        method_name = "Revised simplex"
    else:
        raise ValueError("method_key must be 'solver', 'tableau', or 'revised'.")

    start_time = time.perf_counter()
    simplex_result = solver(A, b, c, basis)
    end_time = time.perf_counter()

    if simplex_result["status"] != "optimal":
        raise RuntimeError(f"{method_name} did not solve dataset '{dataset_name}' to optimality.")

    solve_result = {
        "status": simplex_result["status"],
        "objective": simplex_result["value"],
        "runtime_seconds": end_time - start_time,
        "variable_count": A.shape[1],
        "constraint_count": A.shape[0],
        "iterations": simplex_result["iterations"],
    }
    unpacked = unpack_benchmark_lunar_solution(dataset_name, simplex_result["x"])
    return build_method_result_record(dataset_name, method_name, solve_result, unpacked)


# ~~~~~~~~~~~~~~
# figures

def build_runtime_figure(solver_results, tableau_results, revised_results):
    # Build the runtime plot used in the report. The y-axis is logarithmic
    # because the PuLP/CBC startup overhead is much larger than the custom-method timings.
    label_names = [name.capitalize() for name in DATASET_ORDER]
    load_counts = [DATASETS[name]["report_load_count"] for name in DATASET_ORDER]
    constraint_counts = [tableau_results[i]["constraint_count"] for i in range(len(DATASET_ORDER))]
    solver_runtime = [results["runtime_seconds"] for results in solver_results]
    tableau_runtime = [results["runtime_seconds"] for results in tableau_results]
    revised_runtime = [results["runtime_seconds"] for results in revised_results]
    solver_vars = [results["variable_count"] for results in solver_results]
    simplex_vars = [results["variable_count"] for results in tableau_results]
    tableau_iters = [results["iterations"] for results in tableau_results]
    revised_iters = [results["iterations"] for results in revised_results]

    fig = plt.figure(figsize=(11, 7), dpi=120)
    ax = fig.add_subplot(111)
    x = list(range(len(label_names)))

    ax.plot(x, solver_runtime, marker="o", linewidth=2, label="PuLP/CBC solver")
    ax.plot(x, tableau_runtime, marker="s", linewidth=2, label="Tableau simplex")
    ax.plot(x, revised_runtime, marker="^", linewidth=2, label="Revised simplex")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{name}\n{loads} loads\n{cons} constraints"
        for name, loads, cons in zip(label_names, load_counts, constraint_counts)
    ])
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_title("Runtime Comparison Across Dataset Size\nLunar Base Power Allocation")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    for xi, yi in zip(x, solver_runtime):
        ax.annotate(f"{yi:.6f}s", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center")
    for xi, yi in zip(x, tableau_runtime):
        ax.annotate(f"{yi:.6f}s", (xi, yi), textcoords="offset points", xytext=(0, -16), ha="center")
    for xi, yi in zip(x, revised_runtime):
        ax.annotate(f"{yi:.6f}s", (xi, yi), textcoords="offset points", xytext=(0, 12), ha="center")

    note = (
        "Model size notes:\n"
        f"Solver variables = {solver_vars[0]}, {solver_vars[1]}, {solver_vars[2]}\n"
        f"Simplex variables = {simplex_vars[0]}, {simplex_vars[1]}, {simplex_vars[2]}\n"
        f"Tableau pivots = {tableau_iters[0]}, {tableau_iters[1]}, {tableau_iters[2]}\n"
        f"Revised pivots = {revised_iters[0]}, {revised_iters[1]}, {revised_iters[2]}"
    )
    fig.subplots_adjust(bottom=0.24)
    fig.text(
        0.12,
        0.07,
        note,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "runtime_comparison_figure.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "runtime_comparison_figure.png", bbox_inches="tight")
    return fig


def aggregate_large_case_for_figure(large_result):
    # Combine individual loads into the four categories shown in the report figure.
    delivered = large_result["delivered"]
    shortage = large_result["shortage"]

    delivered_by_category = {
        "Habitat": [
            FIXED_HABITAT_KEEP_ALIVE["crewed"] + delivered["h1"]["crewed"],
            FIXED_HABITAT_KEEP_ALIVE["dormant"] + delivered["h1"]["dormant"],
        ],
        "Mobility": [
            delivered["r1"]["crewed"] + delivered.get("r2", {}).get("crewed", 0.0),
            delivered["r1"]["dormant"] + delivered.get("r2", {}).get("dormant", 0.0),
        ],
        "Science": [
            delivered["s"]["crewed"],
            delivered["s"]["dormant"],
        ],
        "ISRU": [
            delivered.get("ir", {}).get("crewed", 0.0) + delivered.get("ic", {}).get("crewed", 0.0),
            delivered.get("ir", {}).get("dormant", 0.0) + delivered.get("ic", {}).get("dormant", 0.0),
        ],
    }

    shortage_by_category = {
        "Habitat": [shortage["h1"]["crewed"], shortage["h1"]["dormant"]],
        "Mobility": [
            shortage["r1"]["crewed"] + shortage.get("r2", {}).get("crewed", 0.0),
            shortage["r1"]["dormant"] + shortage.get("r2", {}).get("dormant", 0.0),
        ],
        "Science": [shortage["s"]["crewed"], shortage["s"]["dormant"]],
        "ISRU": [
            shortage.get("ir", {}).get("crewed", 0.0) + shortage.get("ic", {}).get("crewed", 0.0),
            shortage.get("ir", {}).get("dormant", 0.0) + shortage.get("ic", {}).get("dormant", 0.0),
        ],
    }

    return delivered_by_category, shortage_by_category


def build_allocation_figure(large_result):
    # Build the large-case allocation figure from the tableau solution.
    delivered_by_category, shortage_by_category = aggregate_large_case_for_figure(large_result)

    fig = plt.figure(figsize=(12, 7), dpi=120)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    x = list(range(2))
    mode_names = ["Crewed", "Dormant"]

    bottom = [0.0, 0.0]
    for label, values in delivered_by_category.items():
        ax1.bar(x, values, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_names)
    ax1.set_ylabel("Delivered power (kW)")
    ax1.set_title("Large-Case Delivered Allocation by Mode")
    ax1.legend()
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

    for xi, total in zip(x, bottom):
        ax1.annotate(
            f"Total delivered = {total:.2f} kW",
            (xi, total),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )

    bottom = [0.0, 0.0]
    for label, values in shortage_by_category.items():
        ax2.bar(x, values, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax2.set_xticks(x)
    ax2.set_xticklabels(mode_names)
    ax2.set_ylabel("Unmet demand (kW)")
    ax2.set_title("Large-Case Shortage by Mode")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)

    for xi, total in zip(x, bottom):
        if total > 0:
            ax2.annotate(
                f"Total shortage = {total:.2f} kW",
                (xi, total),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
            )
        else:
            ax2.annotate("No shortage", (xi, 0), textcoords="offset points", xytext=(0, 6), ha="center")

    fig.suptitle("Allocation by Operating Mode\nLunar Base Power Allocation Large Dataset", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Delivered-power totals are shown after efficiency losses.\n"
        "Crewed mode is scarce; dormant mode remains fully supportable.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "allocation_by_mode_figure.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "allocation_by_mode_figure.png", bbox_inches="tight")
    return fig


# ~~~~~~~~~~~~~~
# part 3: driver

PART3_METHODS = {
    "solver": "PuLP/CBC solver",
    "tableau": "Tableau simplex",
    "revised": "Revised simplex",
}


def run_part3_computational_study():
    print_separator("SECTION 3: PART 3 LUNAR COMPUTATIONAL STUDY")

    all_results = {key: [] for key in PART3_METHODS}

    for dataset_name in DATASET_ORDER:
        for method_key in PART3_METHODS:
            result = solve_lunar_dataset_with_method(dataset_name, method_key)
            all_results[method_key].append(result)
            print_lunar_dataset_results(result)

    print_method_summary("Tableau summary", all_results["tableau"])
    print_method_summary("Solver summary", all_results["solver"])
    print_method_summary("Revised summary", all_results["revised"])

    build_runtime_figure(
        all_results["solver"],
        all_results["tableau"],
        all_results["revised"],
    )
    build_allocation_figure(all_results["tableau"][-1])
    # Disabled for command-line runs.
    # plt.show()


# ~~~~~~~~~~~~~~
# main

def main():
    run_part1_checks()
    run_part2_summary()
    run_part3_computational_study()


if __name__ == "__main__":
    main()