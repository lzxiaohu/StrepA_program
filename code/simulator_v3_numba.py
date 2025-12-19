import numpy as np
import matplotlib.pyplot as plt
import functions_list
import os
import time

start = time.perf_counter()
np.random.seed(40233)

DurationSimulation = 20  #duration of simulation by year
Nstrains = 42  #number of strains 42
Nagents = 2500  #total number of hosts
AgeDeath = 71  #age that all hosts die (years)
BasicReproductionNumber = 2.07
Dimmunity = 0.5 * 52.14  #duration of immunity (weeks)
x = 10  #resistance to co-infection
alpha = 3  #migration rate per week per population
Cperweek = 34.53  #number of contacts per week 34.53
sigma = 1  #strength of strain-specific immunity
omega = 0.1  #strength of cross-strain immunity

params = np.array([DurationSimulation, Nstrains, Dimmunity, sigma, omega, x,
                   Cperweek, Nagents, alpha, AgeDeath, BasicReproductionNumber], dtype=float)
# print(params)
[AgentCharacteristics, ImmuneStatus, _] = functions_list.initialise_agents(params)
# print(AgentCharacteristics)
# print(ImmuneStatus)

[SSPrev,AgentsInfectedByKStrains] = functions_list.simulator_v3_numba(AgentCharacteristics, ImmuneStatus, params, 0, 1)
print(SSPrev)
# np.savetxt("experimental_data/SSPrev_seed40233.csv", SSPrev, delimiter=",", fmt="%.6g")  # fmt for numbers
# np.savetxt("experimental_data/AgentsInfectedByKStrains_seed40233.csv", AgentsInfectedByKStrains, delimiter=",", fmt="%.6g")
# SSPrev.to_csv('/experimental_data/ssPrev_seed3.csv')
# AgentsInfectedByKStrains.to_csv('/experimental_data/agentsInfectedByKStrains_seed3.csv')


end = time.perf_counter()
print(f"Elapsed: {end - start:.4f} s")

ttime = np.linspace(0.0, DurationSimulation, SSPrev.shape[1])

plt.figure()
plt.xlabel("Time (years)")
plt.ylabel("Number of infections")

# MATLAB line(ttime, SSPrev) plots one line per row (strain),
# so transpose for matplotlib (each column vs x).
plt.plot(ttime, SSPrev.T, linewidth=2)

# Optional axis limits (uncomment to mimic your commented line)
# plt.axis([DurationSimulation - 2, DurationSimulation, 0, 500])

plt.grid(True)
plt.title("Strain-specific infections over time")
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ttime, SSPrev.T, linewidth=2)
ax.set_xlabel("Time (years)")
ax.set_ylabel("Number of infections")
ax.set_title("Strain-specific infections over time")
fig.tight_layout()  # avoid clipped labels
os.makedirs(".ipynb_checkpoints/figs", exist_ok=True)
# fig.savefig("figs/strains_over_time_seed40233.png", dpi=300)


coinf = AgentsInfectedByKStrains[:6, -1].astype(float)
total5 = coinf.sum()
prop = coinf / total5 if total5 > 0 else np.zeros_like(coinf)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(np.arange(1, 7), prop)
ax.set_xlabel('Number of infections per host')
ax.set_ylabel('Final host proportion')
ax.set_xlim(0, 7)
ax.set_ylim(0, 1)
ax.set_title('End-of-simulation co-infection distribution')
fig.tight_layout()

# Save the figure
os.makedirs(".ipynb_checkpoints/figs", exist_ok=True)
# fig.savefig("figs/coinfection_distribution_seed40233.png", dpi=300, bbox_inches='tight')

# ---------- 1) Prevalence & Diversity (dual axis) ----------
Diversity = functions_list.div(SSPrev)                                   # (T,)
Prevalence = AgentsInfectedByKStrains.sum(axis=0) / Nagents  # (T,)

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(ttime, Prevalence * 100, color='k', linewidth=2)
ax1.set_ylabel('Prevalence')
ax1.set_xlabel('Time (years)')
ax1.set_xlim(DurationSimulation - 2, DurationSimulation)
ax1.set_ylim(0, 80)

ax2 = ax1.twinx()
ax2.plot(ttime, Diversity, linewidth=2)
ax2.set_ylabel('Diversity')
ax2.set_xlim(DurationSimulation - 2, DurationSimulation)
ax2.set_ylim(0, 40)

fig.tight_layout()
# fig.savefig("figs/prevalence_diversity_seed40233.png", dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)

# ---------- 2) Heat map of extant strains ----------
Outbreak = SSPrev
xm, ym = Outbreak.shape
xx = np.arange(1, xm + 1)          # 1..xm
yy = np.arange(1, ym + 1) / 365.0  # years, like (1:ym)/365

fig = plt.figure(figsize=(7, 4.5))
functions_list.plotheatmap(xx, yy, Outbreak, vmax=90, xylim=(-0.03, 10.005, 0.4, 42.6))
plt.xlabel('Time (years)')
plt.ylabel('Strain number')
plt.tight_layout()
# plt.savefig("figs/heatmap_extant_strains_seed40233.png", dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)

# ---------- 3) 10-year rolling average diversity ----------
T = SSPrev.shape[1]
PrevD = np.zeros_like(SSPrev)
window = int(365 * 10)

for i in range(T):
    start = max(0, i - window + 1)
    PrevD[:, i] = SSPrev[:, start:i+1].sum(axis=1)

DivAvg2 = functions_list.div(PrevD)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ttime, DivAvg2, linewidth=2)
ax.set_ylabel('10-year rolling avg diversity (prev)')
ax.set_xlabel('Time (years)')
fig.tight_layout()
# fig.savefig("figs/diversity_rolling_10y_seed40233.png", dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)