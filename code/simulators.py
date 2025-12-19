#
# Calibrate StrepA ABM with LFIRE using simulation-step indices

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless

import elfi
import pylfire
from numpy.random import default_rng, SeedSequence
from numpy.random import Generator as _NpGen, RandomState as _RS

from pylfire.classifiers.classifier import Classifier
from sklearn.linear_model import LogisticRegressionCV
import hashlib

import functions_list


# 0) Fixed hyperparameters (order MUST match functions_list.parameters())
# parameters(): [DurationSimulation, Nstrains, Dimmunity, sigma, omega, x,
#                Cperweek, Nagents, alpha, AgeDeath, BasicReproductionNumber]
# ----------------------------
DurationSimulation = 20.0       # years
Nstrains = 42
Dimmunity = 0.5 * 52.14         # weeks
omega = 0.1
x = 10.0
Cperweek = 34.53
Nagents = 2500
alpha = 3.0
AgeDeath = 71.0


# R0_LOW,  R0_HIGH  = 0.0, 5.0  # reproductionNumber
# SIG_LOW, SIG_HIGH = 0.5, 1.0  # sigma

# ----------------------------
# 3) Build ABM param vector (theta = [R0, sigma])
# ----------------------------
def build_params(theta2):
    th = np.asarray(theta2, float).ravel()
    if th.size < 2:
        raise ValueError(f"theta2 must be length-2, got {np.shape(theta2)}")
    R0, sigma = float(th[0]), float(th[1])
    return np.array([
        DurationSimulation, Nstrains, Dimmunity, sigma, omega, x,
        Cperweek, Nagents, alpha, AgeDeath, R0
    ], dtype=float)


# ----------------------------
# 4) One ABM run → full prevalence series (length T)
# ----------------------------

def seed_from_theta(theta2, master_seed: int = 123) -> int:
    th = np.asarray(theta2, np.float64).ravel()
    b = th.tobytes() + np.uint64(master_seed).tobytes()
    return int.from_bytes(hashlib.sha1(b).digest()[:8], 'little')


def simulate_prevalence_v5_numba(theta2):
    # derive a child seed from ELFI's rng
    # rng = default_rng(seed)
    seed = seed_from_theta(theta2, master_seed=123)
    rng = default_rng(seed)
    params = build_params(theta2)
    AC, IMM, _ = functions_list.initialise_agents_v5(params, rng=rng)

    # call the reproducible simulator that uses only this seed
    SSPrev_selected, SSPrev, AIBKS = functions_list.simulator_v5_numba(
        AC, IMM, params, 0, 1, seed=seed
    )

    # Option A: return the 42x23 matrix (strain × selected times)
    return SSPrev_selected.astype(float)

# ----------------------------
# 5) Filter indices to those available from the simulator; align y_obs
# ----------------------------
# Dry run to learn T

_Tdry = simulate_prevalence_v5_numba(np.array([2.0, 1.0], float))
T = _Tdry.size
print("T's size", T)
_Tdry1 = simulate_prevalence_v5_numba(np.array([2.0, 1.0], float))
print(np.allclose(_Tdry, _Tdry1), _Tdry.shape == _Tdry1.shape)