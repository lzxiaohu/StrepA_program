# filename: R0-sigma_sensitive.py


# packages:
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import hashlib
import time

import summary_stats_elms as ss
import functions_list


# environment settings:
start = time.perf_counter()


# fixed parameters
DurationSimulation = 20.0     # years
Nstrains = 20       # number of strains
omega = 0.1     # immunity cross strains
x = 10.0        #
Cperweek = 34.53    #
Nagents = 1000      # number of agents
alpha = 3.0         #
AgeDeath = 71.0     #
Dimmunity = 0.4 * 52.14     # weeks
# R0: updated parameter (Basic reproductive number) [1.0, 12.0]
R0_range = [1.0, 12.0]
# Sigma: updated parameter (strain-specific immunity [0, 1])
sigma_range = [0.1, 1.0]
fixed_params = np.array([DurationSimulation, Nstrains, Dimmunity, omega,
                         x, Cperweek, Nagents, alpha,
                         AgeDeath], dtype=float)
rng = np.random.default_rng(123)


# function: build parameters
def build_params(theta, fixed_params):
    theta = np.asarray(theta, float).ravel()
    if theta.size < 2:
        raise ValueError(f"theta must be length-2, got {np.shape(theta)}")
    R0, sigma = float(theta[0]), float(theta[1])
    return np.array([
        fixed_params[0], fixed_params[1], fixed_params[2], sigma,
        fixed_params[3], fixed_params[4], fixed_params[5], fixed_params[6],
        fixed_params[7], fixed_params[8], R0
    ], dtype=float)

# function: seed_from_theta
def seed_from_theta(theta, master_seed: int = 123):
    th = np.asarray(theta, np.float64).ravel()
    b  = th.tobytes() + np.uint64(master_seed).tobytes()
    return int.from_bytes(hashlib.sha1(b).digest()[:8], 'little')


# function: simulate_prevalence_v5_numba
def simulate_prevalence_v5_numba(theta, fixed_params, seed):
    #
    #
    seed = seed_from_theta(theta, master_seed=seed)
    rng = default_rng(seed)
    params = build_params(theta, fixed_params)
    AC, IMM, _ = functions_list.initialise_agents_v5(params, rng=rng)

    # call the reproducible simulator that uses only this seed
    SSPrev_selected, SSPrev, AIBKS = functions_list.simulator_v5_numba(
        AC, IMM, params, 0, 1, seed=seed
    )

    # Option A: return the Nstrain * 23 matrix (strain × selected times)
    return SSPrev_selected.astype(float)


# function: summary_stats()
def summary_stats(series_2d):
    y = np.asarray(series_2d, float).ravel()
    avg_time_obs = ss.avg_time_obs_str(series_2d)
    max_time_obs = ss.max_time_obs_str(series_2d)
    num_strains_obs = ss.num_strains_obs_str(series_2d)
    avg_time_repeat_obs = ss.avg_time_repeat_inf_numpy(series_2d)
    var_time_repeat_obs = ss.var_time_repeat_inf_numpy(series_2d)
    avg_prev_obs = ss.avg_prev_numpy(series_2d)
    var_prev_obs = ss.var_prev_numpy(series_2d)
    avg_div_obs = ss.avg_div_numpy(series_2d)
    var_div_obs = ss.var_div_numpy(series_2d)
    max_abundance_obs = ss.max_abundance_numpy(series_2d)
    avg_npmi_obs = ss.avg_npmi_numpy(series_2d)
    div_all_isolates_obs = ss.div_all_isolates_numpy(series_2d)

    return np.array(
        [avg_time_obs, max_time_obs, num_strains_obs, avg_time_repeat_obs, var_time_repeat_obs, avg_prev_obs,
         var_prev_obs, avg_div_obs, var_div_obs, max_abundance_obs, avg_npmi_obs, div_all_isolates_obs], float)


# test functions:
# synthetic data
_Tdry = simulate_prevalence_v5_numba(np.array([2.0, 0.8], float), fixed_params, seed=int(123))
T = _Tdry.size
print("T's size", T)
_Tdry1 = simulate_prevalence_v5_numba(np.array([2.0, 0.8], float), fixed_params, seed=int(123))
print(np.allclose(_Tdry, _Tdry1), _Tdry.shape == _Tdry1.shape)
s_obs_v5_numba = summary_stats(_Tdry)
# y_obs_array = _Tdry
print("s_obs", s_obs_v5_numba)
# print(_Tdry)

ii = R0_range[0]
ii_R0 = 1.0
jj_sigma = 0.10
rows = []
while ii <= R0_range[1]:

    jj = sigma_range[0]
    while jj <= sigma_range[1]:
        y_sim = simulate_prevalence_v5_numba(np.array([ii, jj], float), fixed_params, seed=int(123))
        s_sim = summary_stats(y_sim)
        row = np.concatenate(([ii, jj], np.asarray(s_sim, float).ravel()))
        rows.append(row)
        jj = jj + jj_sigma

    ii = ii + ii_R0
    print("R0 status", ii)

results = np.vstack(rows)
end = time.perf_counter()
print(f"Elapsed: {end - start:.4f} s")

sum_names = ["avg_time","max_time","num_strains","avg_time_repeat",
             "var_time_repeat", "avg_prev", "var_prev", "avg_div",
             "var_div", "max_abundance", "avg_npmi", "div_all_isolates"
             ]
header = ",".join(["R0","sigma"] + sum_names)
np.savetxt("./experimental_data/R0_sigma_results_v2.csv", results, delimiter=",", header=header, comments="")
