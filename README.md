# StrepA Agent-Based Model (ABM) + Calibration & Sensitivity Analysis

This repository contains a Python implementation of an **agent-based model (ABM)** for **Group A Streptococcus (StrepA)** transmission dynamics and strain structure, plus scripts for **likelihood-free calibration** and **sensitivity analysis**.

The codebase is designed for:
- running stochastic ABM simulations (multiple strains, immunity, migration, etc.),
- extracting **summary statistics** from simulated outputs,
- performing **likelihood-free inference** (LFIRE / ratio estimation) to calibrate key parameters,
- generating **sensitivity analysis datasets** across parameter grids and random seeds,
- visualizing uncertainty across repeated stochastic runs.

---

## Project Structure (typical)

> Filenames may differ slightly depending on your local organization.

- `functions_list*.py`  
  Core ABM implementation and utilities (initialization, simulator variants, numba-accelerated simulators, etc.).

- `summary_stats*.py`  
  Summary statistic functions used for calibration/sensitivity analysis (e.g., prevalence, diversity, persistence, recurrence, NPMI).

- `*_lfire*.py` / notebooks  
  Calibration scripts using **LFIRE (Likelihood-Free Inference by Ratio Estimation)**, typically via `pylfire` + `elfi`.

- `*_sensitive*.py`  
  Sensitivity analysis scripts to evaluate outputs across parameter grids and repeated random seeds.

- `experimental_data/` (optional / local)  
  Output CSVs, posterior grids, and figures generated during experiments.

---

## Model Overview

The StrepA ABM simulates a population of hosts and multiple StrepA strains over discrete time steps. Key modeled processes include:

- **Transmission through contacts**  
  Infection spreads through pairwise contacts; transmission probability is controlled by parameters such as `R0` (basic reproduction number) and contact rate.

- **Recovery and immunity**
  - recovery occurs stochastically,
  - immunity may be strain-specific and/or cross-strain,
  - immunity can wane over time depending on configuration.

- **Co-infection / carrying capacity**
  The model supports multiple concurrent strain infections subject to a co-infection constraint.

- **Migration / importation**
  Hosts may be replaced or new infections imported at a specified migration rate.

Model outputs include time series such as:
- strain prevalence/abundance over time (`SSPrev`),
- distribution of how many strains infect each agent (`AgentsInfectedByKStrains`),
- optionally, selected observation-time outputs to match study design.

---

## Key Parameters (examples)

Some parameters commonly explored during calibration/sensitivity:

- `R0` : basic reproduction number (transmission strength)
- `sigma` : strain-specific immunity strength (or model-specific infection/immunity scaling; see implementation)
- `Dimmunity` : duration of immunity / waning timescale (often in weeks)

Other fixed hyperparameters typically include:
- `Nagents`, `Nstrains`, `Cperweek`, `alpha` (migration), `omega` (cross-immunity), etc.

---

## Calibration (Likelihood-Free Inference)

This repo supports calibration via **LFIRE**:
- simulate ABM outputs under candidate parameters,
- compress outputs into summary statistics,
- train a classifier to approximate the likelihood ratio,
- evaluate an approximate posterior over a parameter grid.

LFIRE is useful when the simulator is stochastic and the likelihood is intractable.

> Depending on your setup, you may need `pylfire` and `elfi`.

---

## Sensitivity Analysis (Repeated Random Seeds)

Sensitivity analysis scripts typically:
1. define a parameter grid over `(R0, sigma)` (and optionally `Dimmunity`),
2. run the stochastic simulator across many random seeds,
3. compute summary statistics for each run,
4. save results as CSV for later correlation analysis and plotting.

For uncertainty visualization, the repo includes plotting utilities such as:
- multi-panel plots of `A1 vs B1` at fixed `A2`,
- mean/median curves with **5–95% bands** across random seeds.

---

## Installation

Create and activate a Python environment (conda recommended):

```bash
conda create -n strepa_env python=3.10 -y
conda activate strepa_env
