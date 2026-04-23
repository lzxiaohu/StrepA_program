# file name: simulator.py


# Packages:
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib
from fontTools.misc.psLib import endofthingRE
# matplotlib.use('Agg')
from numpy.random import default_rng, SeedSequence
from numpy.random import Generator as _NpGen, RandomState as _RS
import functions_list_260305 as functions_list
import summary_stats_elms_260305 as ss
import hashlib
import time, datetime
import logging
import sys

start = time.perf_counter()

core_params_num = 2  # core params: R0 and sigma
# core_params_num = 3  # core params: R0, sigma, and Dimmunity

# fixed parameters
DurationSimulation = 20.0  # years: 20.0
Nstrains = 40  # number of strains: 42
omega = 0.2  # immunity cross strains: 0.1
x = 10.0  # Resistance to co-infection
Cperweek = 34.53  #
Nagents = 1000  # number of agents
alpha = 7.55  # Migration rate per week in population (0.007 per week per person)
AgeDeath = 71.0  # life expectancy
# R0: updated parameter (Basic reproductive number)
# Sigma: updated parameter ()

if core_params_num == 2:
    Dimmunity = 10.0 * 52.14  # weeks: 0.5 * 52.14
    fixed_params = np.array([DurationSimulation, Nstrains, Dimmunity, omega,
                             x, Cperweek, Nagents, alpha,
                             AgeDeath], dtype=float)
elif core_params_num == 3:
    fixed_params = np.array([DurationSimulation, Nstrains, omega, x,
                             Cperweek, Nagents, alpha, AgeDeath], dtype=float)
else:
    raise ValueError('Invalid core params num')

rng = np.random.default_rng(123)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout  # Important for nohup
)


# function: build parameters
def build_params(theta, fixed_params, core_params_num):
    theta = np.asarray(theta, float).ravel()
    if theta.size != core_params_num:
        raise ValueError(f"theta must be length-{core_params_num}, got {np.shape(theta)}")
    if core_params_num == 2:
        R0, sigma = float(theta[0]), float(theta[1])
        return np.array([fixed_params[0], fixed_params[1], fixed_params[2], sigma,
                         fixed_params[3], fixed_params[4], fixed_params[5], fixed_params[6],
                         fixed_params[7], fixed_params[8], R0
                         ], dtype=float)
    elif core_params_num == 3:
        R0, sigma, Dimmunity = float(theta[0]), float(theta[1]), float(theta[2])
        return np.array([fixed_params[0], fixed_params[1], Dimmunity, sigma,
                         fixed_params[2], fixed_params[3], fixed_params[4], fixed_params[5],
                         fixed_params[6], fixed_params[7], R0
                         ], dtype=float)
    else:
        raise ValueError('Invalid core params num')


# function: seed_from_theta
def seed_from_theta(theta, master_seed: int = 123):
    th = np.asarray(theta, np.float64).ravel()
    b = th.tobytes() + np.uint64(master_seed).tobytes()
    return int.from_bytes(hashlib.sha1(b).digest()[:8], 'little')


# function: simulate_prevalence_v5_numba
def simulate_prevalence_v5_numba(theta, fixed_params, core_params_num, seed):
    #
    #
    seed = seed_from_theta(theta, master_seed=seed)
    rng = default_rng(seed)
    params = build_params(theta, fixed_params, core_params_num)
    AC, IMM, _ = functions_list.initialise_agents_v5(params, rng=rng)

    # call the reproducible simulator that uses only this seed
    SSPrev_selected, SSPrev, AIBKS = functions_list.simulator_v5_numba(
        AC, IMM, params, 0, 1, seed=seed
    )

    # Option A: return the Nstrain * 23 matrix (strain × selected times)
    return SSPrev_selected.astype(float)


# function: summary_stats()
def summary_stats(series_2d, scale=None):
    """
    Compute summary statistics, replacing NaN with scale values if provided.

    Parameters:
    -----------
    series_2d : array-like
        Input data (strains × timepoints)
    scale : array-like, optional
        Scale values to use for NaN replacement [avg_prev, var_prev, avg_npmi, div]
        Default: [520.0, 56700.0, 0.448, 26.4]

    Returns:
    --------
    stats : array, shape (4,)
        [avg_prev, var_prev, avg_npmi, diversity]
    """

    # y = np.asarray(series_2d, float).ravel()
    # avg_time_obs = ss.avg_time_obs_str(series_2d)
    # max_time_obs = ss.max_time_obs_str(series_2d)
    # num_strains_obs = ss.num_strains_obs_str(series_2d)
    # avg_time_repeat_obs = ss.avg_time_repeat_inf_numpy(series_2d)
    # var_time_repeat_obs = ss.var_time_repeat_inf_numpy(series_2d)
    avg_prev_obs = ss.avg_prev_numpy(series_2d)
    var_prev_obs = np.sqrt(ss.var_prev_numpy(series_2d))
    # avg_div_obs = ss.avg_div_numpy(series_2d)
    # var_div_obs = ss.var_div_numpy(series_2d)
    # max_abundance_obs = ss.max_abundance_numpy(series_2d)
    avg_npmi_obs = ss.avg_npmi_numpy(series_2d)
    div_all_isolates_obs = ss.div_all_isolates_numpy(series_2d)
    # print("s_obs: ", avg_time_obs, avg_prev_obs, var_prev_obs, avg_div_obs, avg_npmi_obs)

    # Combine into array
    stats = np.array(
        [avg_prev_obs, var_prev_obs, avg_npmi_obs, div_all_isolates_obs],
        dtype=float
    )

    # Replace NaN with scale values if provided
    if scale is not None:
        scale = np.asarray(scale, dtype=float)
        nan_mask = np.isnan(stats)
        stats[nan_mask] = scale[nan_mask]

    return stats


# synthetic data
if core_params_num == 2:
    _Tdry = simulate_prevalence_v5_numba(np.array([2.07, 0.8], float), fixed_params, core_params_num, seed=int(123))
    T = _Tdry.size
    logging.info(f"T's size: {T}")
    print("T's size", _Tdry)
    # _Tdry1 = simulate_prevalence_v5_numba(np.array([2.07, 0.8], float), fixed_params, core_params_num, seed=int(123))
    # logging.info(f"allclose={np.allclose(_Tdry, _Tdry1)}, same_shape={_Tdry.shape == _Tdry1.shape}")
    # print(np.allclose(_Tdry, _Tdry1), _Tdry.shape == _Tdry1.shape)
elif core_params_num == 3:
    _Tdry = simulate_prevalence_v5_numba(np.array([2.07, 0.8, 10.0 * 52.14], float), fixed_params, core_params_num,
                                         seed=int(123))
    T = _Tdry.size
    logging.info(f"T's size: {T}")
    # print("T's size", T)
    _Tdry1 = simulate_prevalence_v5_numba(np.array([2.07, 0.8, 10.0 * 52.14], float), fixed_params, core_params_num,
                                          seed=int(123))
    logging.info(f"allclose={np.allclose(_Tdry, _Tdry1)}, same_shape={_Tdry.shape == _Tdry1.shape}")
    # print(np.allclose(_Tdry, _Tdry1), _Tdry.shape == _Tdry1.shape)

else:
    raise ValueError('Invalid core params num')


def plot_strain_infections(Tdry):
    """
    Plot number of infections of each strain as function of time.

    Parameters:
    -----------
    Tdry : ndarray
        40x23 matrix where rows are strains and columns are time points
    """
    plt.figure(figsize=(12, 6))

    # Plot each strain
    for i in range(40):
        plt.plot(Tdry[i, :], alpha=0.6, linewidth=1.5)

    plt.xlabel('Time Point')
    plt.ylabel('Number of Infections')
    plt.title('Strain Infections Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_strain_infections(_Tdry)