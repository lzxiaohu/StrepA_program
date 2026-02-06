# ***** File: ABC_diy.py *****


# *** Packages ***
import numpy as np
import matplotlib.pyplot as plt


# inputs: x_age, y_dis
x_age = [19, 20, 21, 22, 22,
     23, 24, 25, 26, 27,
     29, 30, 35, 40, 50,
     60, 63, 64, 68]
y_dis = [510, 580, 560, 480, 500,
     490, 570, 520, 480, 410,
     430, 420, 390, 450, 380,
     330, 410, 420, 430]
x_age = np.asarray(x_age, dtype=float)
y_dis = np.asarray(y_dis, dtype=float)
data_len = len(x_age)
rng = np.random.default_rng(42)


# function: simulator_y
def simulator_y(x, a, b, sigma, rng):
    # generate dis(tance) by the equation a + b * x + Gaussian noise
    return a + b * x + rng.normal(0.0, sigma, size=len(x))


# function: summary_stats_elm_ab
def summary_stats_elm_ab(x, y):
    # define the elements of summary statistics: intercept a and slope b
    x_bar, y_bar = x.mean(), y.mean()
    b = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
    a = y_bar - b * x_bar
    return a, b


# function: summary_stats
def summary_stats(x, y):
    a_hat, b_hat = summary_stats_elm_ab(x, y)
    return np.array([y.mean(),
                     y.std(ddof=1),
                     a_hat,
                     b_hat])
# s_obs: summary stats of observations
s_obs = summary_stats(x_age, y_dis)
print("summary statistics: ", s_obs)


# scale summaries so no single stat dominates distance
scale = np.array([
    max(y_dis.std(ddof=1), 1e-6),
    max(y_dis.std(ddof=1), 1e-6),
    max(abs(s_obs[2]), 1.0),
    max(abs(s_obs[3]), 1.0),
])
print("scale: ", scale)


# function: discrepancy
def discrepancy(s_sim, s_obs, scale):
    # scale the difference between simulated data and observations
    z = (s_sim - s_obs) / scale
    return np.sqrt(np.sum(z**2))

# function: prior_value
def prior_value(x, y, rng):
    # intercept a: around distance range
    a = rng.uniform(y.min()-50, y.max()+50)
    # slope b: often negative
    b = rng.uniform(-10.0, 0.0)

    # noise signma
    sigma = rng.uniform(1.0, max(5.0, y.std(ddof=1)*2))
    return a, b, sigma


# function: select_epsilon
def select_epsilon(x, y, s_obs, scale, n_pilot=5000, quantile=0.02, rng=None):
    # select an appropriate epsilon
    rng = rng
    dists = np.empty(n_pilot)

    for ii in range(n_pilot):
        a, b, sigma = prior_value(x, y, rng)
        y_sim = simulator_y(x, a, b, sigma, rng)
        s_sim = summary_stats(x, y_sim)
        dists[ii] = discrepancy(s_sim, s_obs, scale)

    eps = np.quantile(dists, quantile)
    return eps, dists

eps, pilots = select_epsilon(x_age, y_dis, s_obs, scale, n_pilot=8000, quantile=0.02, rng=rng)
print("eps: ", eps)


# function: abc_rejct
def abc_reject(x, y, s_obs, scale, eps, n_accept=2000, max_trials=2_000_000, rng=None):
    rng = rng
    accepted = []
    dists_acc = []

    trials = 0

    while len(accepted) < n_accept and trials < max_trials:
        trials += 1
        a, b, sigma = prior_value(x, y, rng)
        y_sim = simulator_y(x, a, b, sigma, rng)
        s_sim = summary_stats(x, y_sim)
        dist = discrepancy(s_sim, s_obs, scale)

        if dist < eps:
            accepted.append((a, b, sigma))
            dists_acc.append(dist)

    acc = np.array(accepted)
    dists_acc = np.array(dists_acc)
    return acc, dists_acc, trials

post, dists_acc, trials = abc_reject(x_age, y_dis, s_obs, scale, eps, n_accept=1500, max_trials=2_000_000, rng=rng)
print("Accepted: ", len(post), "Trials: ", trials, "Acceptance rate: ", len(post)/trials)

a_samps, b_samps, sigma_samps = post[:, 0], post[:, 1], post[:, 2]
print("Posterior mean b: ", b_samps.mean(), "  P(b<0):", np.mean(b_samps < 0))

plt.hist(b_samps, bins=30, density=True)
plt.title("ABC posterior of slope b")
plt.xlabel("b")
plt.ylabel("Density")
plt.show()

new_age = 45.0
y_new = a_samps + b_samps * new_age + rng.normal(0.0, sigma_samps, size=len(sigma_samps))

plt.hist(y_new, bins=30, density=True)
plt.title(f"ABC posterior predictive at age={new_age} years old")
plt.xlabel("Predicted distance")
plt.ylabel("Density")
plt.show()
print("Predictive mean: ", y_new.mean())
print("95% predictive interval: ", np.quantile(y_new, [0.025, 0.975]))

x_grid = np.linspace(x_age.min(), x_age.max(), 200)
idx = rng.choice(len(a_samps), size=min(200, len(a_samps)), replace=False)
plt.scatter(x_age, y_dis, label="Observed data")

for ii in idx:
    plt.plot(x_grid, a_samps[ii] + b_samps[ii] * x_grid, alpha=0.5)

plt.xlabel("Age")
plt.ylabel("Max legible distance")
plt.title("ABC posterior regression lines")
plt.show()