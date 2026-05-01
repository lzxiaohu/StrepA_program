# file name: R0_sigma_2params_sensitive_update.py

import os
import time
import hashlib
import numpy as np
from numpy.random import default_rng, SeedSequence

import summary_stats_elms_260305 as ss
import functions_list_260305 as functions_list

start = time.perf_counter()

# ----------------------
# settings
# ----------------------
core_params_num = 2  # 2: (R0, sigma)   3: (R0, sigma, Dimmunity)
R0_vals = np.arange(1.0, 12.0 + 1e-12, 1.0)          # 12 values
sigma_vals = np.round(np.arange(0.1, 1.0 + 1e-12, 0.1), 10)  # 10 values
Dimmunity_vals = np.round(np.arange(0.05, 0.50 + 1e-12, 0.05), 10)  # 10 values

# fixed parameters
Dimmunity = 10.0 * 52.14
DurationSimulation = 20
Nstrains = 40
omega = 0.2
x = 10.0
Cperweek = 34.53
Nagents = 1000
alpha = 7.0
AgeDeath = 71.0

if core_params_num == 2:
    fixed_params = np.array([DurationSimulation, Nstrains, Dimmunity, omega,
                         x, Cperweek, Nagents, alpha,
                         AgeDeath], dtype=float)
elif core_params_num == 3:
    fixed_params = np.array([DurationSimulation, Nstrains, omega,x,
                             Cperweek, Nagents, alpha, AgeDeath], dtype=float)
else:
    raise ValueError('Invalid core params num')

# output
out_dir = "../../experimental_data/from_260430"
os.makedirs(out_dir, exist_ok=True)

# how many random-seed repeats
N_REPS = 100
# MASTER_SEED: [66316748, 2930678936, 2546691362, 231159514, 3904498325,
#               946438445, 1095601156, 791870896, 1432871125, 755510091,
#               1493800520, 3487919346, 1938714511, 3965736568, 1930440936,
#               1187877992, 3387705611, 3520819031, 3701866991, 3822060012]
MASTER_SEED = 66316748  # controls reproducibility of the entire experiment


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
def seed_from_theta(theta, master_seed: int):
    th = np.asarray(theta, np.float64).ravel()
    b  = th.tobytes() + np.uint64(master_seed).tobytes()
    return int.from_bytes(hashlib.sha1(b).digest()[:8], 'little')


# function: simulate_prevalence_v5_numba
def simulate_prevalence_v5_numba(theta, fixed_params, core_params_num, master_seed_for_run: int):
    # deterministic per (theta, run_seed)
    #
    seed = seed_from_theta(theta, master_seed=master_seed_for_run)
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


sum_names = ["avg_time","max_time","num_strains","avg_time_repeat",
             "var_time_repeat","avg_prev","var_prev","avg_div",
             "var_div","max_abundance","avg_npmi","div_all_isolates"]

# --------------------------
# mini test
# ---------------------------
# synthetic data
if core_params_num == 2:
    theta = np.array([5.0, 0.5], float)
elif core_params_num == 3:
    theta = np.array([5.0, 0.5, 0.25 * 52.14], float)
else:
    raise ValueError('Invalid core params num')

y1 = simulate_prevalence_v5_numba(theta, fixed_params, core_params_num, master_seed_for_run=123)
y2 = simulate_prevalence_v5_numba(theta, fixed_params, core_params_num, master_seed_for_run=123)
y3 = simulate_prevalence_v5_numba(theta, fixed_params, core_params_num, master_seed_for_run=222)
print(np.allclose(y1, y2), y1.shape == y2.shape)
print(np.allclose(y1, y3), y1.shape == y3.shape)

# print("s_obs",y1)


# ----------------------------
# SeedSequence spawn 2000 independent streams
# ----------------------------
ss_master = SeedSequence(MASTER_SEED)
children = ss_master.spawn(N_REPS)

# Turn each child into a single integer "run seed"
run_seeds = np.array(
    [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children],
    dtype=np.uint32
)

print(f"Generated {N_REPS} independent run seeds.")

# ----------------------------
# Run sensitivity with random seeds
# ----------------------------
rows = []

for rep, run_seed in enumerate(run_seeds):
    run_seed = int(run_seed)

    if core_params_num == 2:
        for R0 in R0_vals:
            for sigma in sigma_vals:
                theta = np.array([R0, sigma], float)
                y_sim = simulate_prevalence_v5_numba(theta, fixed_params, core_params_num, master_seed_for_run=run_seed)
                s_sim = summary_stats(y_sim)
                row = np.concatenate(([rep, run_seed, R0, sigma], s_sim.ravel()))
                rows.append(row)

    else:  # core_params_num == 3
        for R0 in R0_vals:
            for sigma in sigma_vals:
                for Dim in Dimmunity_vals:
                    theta = np.array([R0, sigma, Dim * 52.14], float)
                    y_sim = simulate_prevalence_v5_numba(theta, fixed_params, core_params_num, master_seed_for_run=run_seed)
                    s_sim = summary_stats(y_sim)
                    row = np.concatenate(([rep, run_seed, R0, sigma, Dim], s_sim.ravel()))
                    rows.append(row)

    if (rep + 1) % 2 == 0:
        print(f"Finished rep {rep+1}/{N_REPS}")

results = np.vstack(rows)

end = time.perf_counter()
print(f"Elapsed: {end - start:.2f} s")

# ----------------------------
# Save
# ----------------------------
if core_params_num == 2:
    header = ",".join(["rep","run_seed","R0","sigma"] + sum_names)
    out_path = os.path.join(out_dir, "R0_sigma_sensitive_2params_batch00.csv")
else:
    header = ",".join(["rep","run_seed","R0","sigma","Dimmunity"] + sum_names)
    out_path = os.path.join(out_dir, "R0_sigma_Dimmunity_sensitive_3params_2000reps.csv")

np.savetxt(out_path, results, delimiter=",", header=header, comments="")
print("Saved:", out_path)