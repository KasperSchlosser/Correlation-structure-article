import pathlib


import numpy as np
import matplotlib.pyplot as plt
import nabqra.plotting as nplt
import scipy.stats as stats
import seaborn as sns


PATH = pathlib.Path.cwd().parents[1]
save_path = PATH / "Results" / "Pseudoresiduals"

plt.close("all")

rng = np.random.RandomState(42)

# %% correlation structure


N = 100000
ar1 = 0.99

sigma = np.sqrt(1 - ar1**2)
s = 2
dist = stats.loguniform(a=1, b=100)


pseudo = np.zeros(N)

res = stats.norm(scale=sigma).rvs(N, random_state=rng)


for i in range(1, N):
    pseudo[i] = ar1 * pseudo[i - 1] + +res[i]


cdf_res = stats.norm().cdf(pseudo)
org_res = dist.ppf(cdf_res)

X = np.linspace(org_res.min(), org_res.max(), 10000)

fig, ax = plt.subplots()

sns.histplot(
    x=org_res,
    stat="density",
    bins="auto",
    ax=ax,
)
ax.plot(X, dist.pdf(X), color="C1", label="Log Uniform(1,100)")
ax.legend()
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim([1, 5])

fig.savefig(save_path / "Figures" / "Marginal distribution")
# plt.close(fig)


fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

axes[0].plot(org_res)
axes[1].plot(cdf_res)
axes[2].plot(pseudo)


axes[0].set_title("Original Space")
axes[1].set_title("CDF Space")
axes[2].set_title("Normal Space")

axes[0].set_xlim([52400, 54000])
# axes[0].set_ylim([0, 5])
axes[2].set_ylim([-3, 3])

fig.savefig(save_path / "Figures" / "Correlation example")
# plt.close(fig)


# %% model checking

dist1 = stats.norm()
dist2 = stats.t(2, loc=0.2)
K = 10000

res = dist1.rvs(K, random_state=rng)
cdf_res = dist1.cdf(res)

X = np.linspace(-4, 4, 1000)

fig, ax = plt.subplots()
ax.hist(res, bins="auto", density=True)
ax.plot(X, dist1.pdf(X), label="Normal Distribution")
ax.plot(X, dist2.pdf(X), label="t Distribution")
ax.legend()
fig.savefig(save_path / "Figures" / "Distributions")


nplt.diagnostic_plots(
    stats.norm().ppf(dist1.cdf(res)),
    np.arange(K),
    save_path=save_path / "Figures" / "normal resid",
    closefig=True,
)
nplt.diagnostic_plots(
    stats.norm().ppf(dist2.cdf(res)),
    np.arange(K),
    save_path=save_path / "Figures" / "t resid",
    closefig=True,
)
