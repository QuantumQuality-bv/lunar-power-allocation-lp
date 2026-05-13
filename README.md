# Lunar Power Allocation LP

This repository contains a Math 464 final project script for modeling lunar base power allocation as a linear programming problem. The script compares two custom simplex implementations against a PuLP/CBC solver baseline across small, medium, and large datasets.

## What the script does

- Verifies tableau simplex and revised simplex implementations on small general-form LP examples.
- Builds nested lunar power-allocation datasets with crewed and dormant operating modes.
- Solves each dataset with tableau simplex, revised simplex, and PuLP/CBC.
- Prints objective values, runtimes, allocation values, shortages, and model-size summaries.
- Saves the runtime and allocation figures to the `figures/` directory.

## Repository structure

```text
lunar-power-allocation-lp/
├── README.md
├── lunar_power_allocation.py
├── requirements.txt
└── figures/
    └── .gitkeep
```

## Setup

Create a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

```bash
python lunar_power_allocation.py
```

The script prints the verification checks and computational-study results to the terminal. Generated plots are saved in:

```text
figures/runtime_comparison_figure.png
figures/runtime_comparison_figure.pdf
figures/allocation_by_mode_figure.png
figures/allocation_by_mode_figure.pdf
```

## Dependencies

The project uses NumPy for matrix operations, PuLP/CBC for the solver baseline, and Matplotlib for figures.
