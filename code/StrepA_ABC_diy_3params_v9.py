# file name: StrepA_ABC_diy_3params_v9.py


# Packages:
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
from numpy.random import default_rng, SeedSequence
from numpy.random import Generator as _NpGen, RandomState as _RS
import functions_list
import summary_stats_elms as ss
import hashlib
import time

start = time.perf_counter()
# fixed parameters
DurationSimulation = 20.0     # years
Nstrains = 20       # number of strains
# Dimmunity = 0.5 * 52.14     # weeks
omega = 0.1     # immunity cross strains
x = 10.0        #
Cperweek = 34.53    #
Nagents = 1000      # number of agents
alpha = 3.0         #
AgeDeath = 71.0     #
# R0: updated parameter (Basic reproductive number)
# Sigma: updated parameter ()
fixed_params = np.array([DurationSimulation, Nstrains, omega,
                         x, Cperweek, Nagents, alpha,
                         AgeDeath], dtype=float)
rng = np.random.default_rng(123)
# function: build parameters
def build_params(theta, fixed_params):
    theta = np.asarray(theta, float).ravel()
    if theta.size < 3:
        raise ValueError(f"theta must be length-2, got {np.shape(theta)}")
    R0, sigma, Dimmunity = float(theta[0]), float(theta[1]), float(theta[2])
    return np.array([
        fixed_params[0], fixed_params[1], Dimmunity, sigma,
        fixed_params[2], fixed_params[3], fixed_params[4], fixed_params[5],
        fixed_params[6], fixed_params[7], R0
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

# synthetic data
_Tdry = simulate_prevalence_v5_numba(np.array([2.0, 0.8, 0.4*52.14], float), fixed_params, seed=int(123))
T = _Tdry.size
print("T's size", T)
_Tdry1 = simulate_prevalence_v5_numba(np.array([2.0, 0.8, 0.4*52.14], float), fixed_params, seed=int(123))
print(np.allclose(_Tdry, _Tdry1), _Tdry.shape == _Tdry1.shape)
# print(_Tdry)

# function: summary_stats()
def summary_stats(series_2d):
    y = np.asarray(series_2d, float).ravel()
    avg_time_obs = ss.avg_time_obs_str(series_2d)
    # max_time_obs = ss.max_time_obs_str(series_2d)
    num_strains_obs = ss.num_strains_obs_str(series_2d)
    # avg_time_repeat_obs = ss.avg_time_repeat_inf_numpy(series_2d)
    # var_time_repeat_obs = ss.var_time_repeat_inf_numpy(series_2d)
    avg_prev_obs = ss.avg_prev_numpy(series_2d)
    # var_prev_obs = np.sqrt(ss.var_prev_numpy(series_2d))
    avg_div_obs = ss.avg_div_numpy(series_2d)
    # var_div_obs = ss.var_div_numpy(series_2d)
    # max_abundance_obs = ss.max_abundance_numpy(series_2d)
    avg_npmi_obs = ss.avg_npmi_numpy(series_2d)
    div_all_isolates_obs = ss.div_all_isolates_numpy(series_2d)
    # print("s_obs: ", avg_time_obs, avg_prev_obs, var_prev_obs, avg_div_obs, avg_npmi_obs)

    return np.array(
        [avg_time_obs, num_strains_obs, avg_prev_obs, avg_div_obs,
         avg_npmi_obs, div_all_isolates_obs], float)


s_obs_v5_numba = summary_stats(_Tdry)
y_obs_array = _Tdry
print("s_obs", s_obs_v5_numba)

# scale = abs(s_obs_v5_numba)
scale = np.array([21.0, 19.0, 250.0, 17, 0.6, 17], dtype=float)
print("scale", scale)

# function: discrepancy
def discrepancy(s_sim, s_obs, scale):
    # scale the difference between simulated data and observations
    z = (s_sim - s_obs) / scale
    return np.sqrt(np.sum(z**2))

# function: prior_value
def prior_value(R0_range, sigma_range, Dimmunity_range, rng):
    # intercept a: around distance range
    R0_sel = rng.uniform(R0_range[0], R0_range[1])
    # slope b: often negative
    sigma_sel = rng.uniform(sigma_range[0], sigma_range[1])
    Dimmunity_sel = rng.uniform(Dimmunity_range[0], Dimmunity_range[1])

    # noise signma
    # sigma = rng.uniform(1.0, max(5.0, y.std(ddof=1)*2))
    return R0_sel, sigma_sel, Dimmunity_sel

# function: select_epsilon
def select_epsilon(R0_range, sigma_range, Dimmunity_range, s_obs, scale, n_pilot=5000, quantile=0.02, seed=123):
    # select an appropriate epsilon
    # rng = rng

    dists = []
    rng = np.random.default_rng(123)

    for ii in range(n_pilot):
        R0_sel, sigma_sel, Dimmunity_sel = prior_value(R0_range, sigma_range, Dimmunity_range, rng)
        y_sim = simulate_prevalence_v5_numba([R0_sel, sigma_sel, Dimmunity_sel*52.14], fixed_params, seed)
        s_sim = summary_stats(y_sim)
        tempt = discrepancy(s_sim, s_obs, scale)
        if np.isnan(tempt):
            # print("this is nan")
            pass
        else:
            dists.append(tempt)
            # print("dist", tempt)
        if ii % 50 == 0:
            print("dist", tempt, "number: ", ii)

    eps = np.quantile(dists, quantile)
    return eps, dists
R0_range= [1.0, 12.0]
sigma_range = [0.1, 1.0]
Dimmunity_range = [0.05, 0.5]
eps, pilots = select_epsilon(R0_range, sigma_range, Dimmunity_range, s_obs_v5_numba, scale, n_pilot=500, quantile=0.2, seed=123)
# eps =0.3252211564759101
print("eps: ", eps)
print("dists: ", len(pilots))
# plt.hist(pilots, bins=30, density=True)
# plt.title("ABC posterior of pilots")
# plt.xlabel("dists")
# plt.ylabel("Density")
# plt.show()


# function: abc_rejct
def abc_reject(R0_range, sigma_range, Dimmunity_range, s_obs, scale, eps, n_accept=2000, max_trials=2_000_000, seed=123):
    rng = np.random.default_rng(123)
    accepted = []
    ss = []
    dists_acc = []

    trials = 0
    count = 0

    while len(accepted) < n_accept and trials < max_trials:
        trials += 1
        R0_sel, sigma_sel, Dimmunity_sel = prior_value(R0_range, sigma_range, Dimmunity_range, rng)
        y_sim = simulate_prevalence_v5_numba([R0_sel, sigma_sel, Dimmunity_sel*52.14], fixed_params, seed)
        s_sim = summary_stats(y_sim)
        dist = discrepancy(s_sim, s_obs, scale)

        if dist < eps:
            count += 1
            accepted.append((R0_sel, sigma_sel, Dimmunity_sel))
            ss.append(s_sim)
            # print("ss:", ss)
            dists_acc.append(dist)

            if count % 30 == 0:
                print("dist", dist, "accepted: ", count, "trials: ", trials)
            else:
                pass

    acc = np.array(accepted)
    dists_acc = np.array(dists_acc)
    ss = np.vstack(ss)
    return acc, dists_acc, trials, ss


post, dists_acc, trials, ss = abc_reject(R0_range, sigma_range, Dimmunity_range, s_obs_v5_numba, scale, eps, n_accept=400, max_trials=2_000_000, seed=123)
print("Accepted: ", len(post), "Trials: ", trials, "Acceptance rate: ", len(post)/trials)

R0_samps, sigma_samps, Dimmunity_samps = post[:, 0], post[:, 1], post[:, 2]
print("Posterior mean R0: ", R0_samps.mean())
print("Posterior mean sigma: ", sigma_samps.mean())
print("Posterior mean Dimmunity: ", Dimmunity_samps.mean())
end = time.perf_counter()
print(f"Elapsed: {end - start:.4f} s")
# print("R0: ",  R0_samps)
# print("sigma: ", sigma_samps)
np.savetxt("experimental_data/R0_samps_3params_3stars_v9_ud3.csv", R0_samps, delimiter=",")
np.savetxt("experimental_data/sigma_samps_3params_3stars_v9_ud3.csv", sigma_samps, delimiter=",")
np.savetxt("experimental_data/Dimmunity_samps_3params_3stars_v9_ud3.csv", Dimmunity_samps, delimiter=",")
np.savetxt("experimental_data/dists_acc_3params_3stars_v9_ud3.csv", dists_acc, delimiter=",")
np.savetxt("experimental_data/ss_3params_3stars_v9_ud3.csv", ss, delimiter=",")
# plt.hist(R0_samps, bins=30, density=True)
# plt.title("ABC posterior of R0")
# plt.xlabel("R0")
# plt.ylabel("Density")
# plt.show()


# plt.hist(sigma_samps, bins=30, density=True)
# plt.title("ABC posterior of sigma")
# plt.xlabel("sigma")
# plt.ylabel("Density")
# plt.show()