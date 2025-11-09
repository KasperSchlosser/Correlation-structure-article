import pathlib
import tomllib

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import properscoring as ps
import seaborn as sns

import nabqra


PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Evaluation"
save_path = PATH / "Results" / "Evaluation"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)

rng = np.random.default_rng(42)

plt.close("all")


# %% distance metrics

dist1 = stats.beta(2, 2, loc=-3, scale=6)
dist2 = stats.beta(7, 11, loc=-3, scale=6)
X = np.linspace(-3, 3, num=100000)


# distributions for comparison
fig, ax = plt.subplots()

lines = ax.plot(X, np.array([dist1.pdf(X), dist2.pdf(X)]).T)
ax.fill_between(X, dist1.pdf(X), color=lines[0].get_color(), alpha=0.5, label=r"$x \sim F = Beta(2,2; -3,3)$")
ax.fill_between(X, dist2.pdf(X), color=lines[1].get_color(), alpha=0.5, label=r"$x \sim G = Beta(7,11; -3,3)$")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("p(x)")

fig.savefig(save_path / "Figures" / "Example Dists")
plt.close(fig)

# %% wasserstein + lp distance
fig, ax = plt.subplots()

ax.plot(X, np.array([dist1.ppf(X), dist2.ppf(X)]).T)
ax.fill_between(X, dist1.ppf(X), dist2.ppf(X), color="grey", alpha=0.3)
# ax.set_ylim([0, 1])
ax.legend(["$F = Beta(2,2; -3,3)$", "$G = Beta(7,11; -3,3)$", "Wasserstein Area"])
ax.set_ylabel("Quantile(q)")
ax.set_xlabel("q")

fig.savefig(save_path / "Figures" / "Wasserstein Area")
plt.close(fig)

fig, ax = plt.subplots()

wasser_table = pd.DataFrame(index=["$W_p$"])
Q = np.linspace(1e-12, 1 - 1e-12, num=10000)

for p in [0.1, 0.5, 1, 2]:
    d = np.abs(dist1.ppf(Q) - dist2.ppf(Q)) ** p

    score = nabqra.scoring.continous_wasserstein(dist1.ppf, dist2.ppf, order=p, lims=[1e-8, 1 - 1e-8], limit=500)[0]
    wasser_table[f"{p:.1f}"] = score

    nabqra.plotting.band_plot(Q, d, d, band_label=f"$ p = {p} $", ax=ax)


ax.set_xlabel("q")
ax.set_ylabel(r"$|F^{-1}(q) - G^{-1}(q)|^p$")
# ax.set_ylim([0, 3])
ax.set_xlim([0, 1])
ax.legend()

fig.savefig(save_path / "Figures" / "Wasserstein Distance")
plt.close(fig)


wasser_table = pd.concat([wasser_table], axis=1, keys=["p"])
wasser_table.style.format(precision=2).to_latex(
    save_path / "Tables" / "Wasserstein.tex",
    hrules=True,
    clines="skip-last;data",
    convert_css=True,
    position="ht",
    position_float="centering",
    multicol_align="c",
    multirow_align="r",
    label="tab:evaluation:wasserstein",
    caption=(
        "Wasserstein distance for the two example distributions.",
        "Wasserstein distance",
    ),
)

# %% kl Divergence

X = np.linspace(-3, 3, num=10000)
X2 = np.linspace(-3, 3, num=10000)

fig, ax = plt.subplots()

a1 = ax.fill_between(X, np.log(dist1.pdf(X)) - np.log(dist2.pdf(X)), color="C0", label=r"$\ln{\frac{f(x)}{g(x)}}$")
a2 = ax.fill_between(X, np.log(dist2.pdf(X)) - np.log(dist1.pdf(X)), color="C1", label=r"$\ln{\frac{g(x)}{f(x)}}$")

ax.set_xlabel("x")
ax.set_ylabel("Excess Information")
ax.set_ylim([-3, 3])
ax.legend()

fig.savefig(save_path / "Figures" / "Excess Information")
plt.close(fig)


fig, ax = plt.subplots()
kl1 = dist1.pdf(X2) * np.log(dist1.pdf(X2) / dist2.pdf(X2))
kl2 = dist2.pdf(X2) * np.log(dist2.pdf(X2) / dist1.pdf(X2))
ax.fill_between(X2, kl1, label=r"$f(x) \ln{\frac{f(x)}{g(x)}}$")
ax.fill_between(X2, kl2, label=r"$g(x) \ln{\frac{g(x)}{f(x)}}$")
ax.plot(X2, kl1 + kl2, color="black", label=r"Amplitude Difference")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("KL-Divergence")

fig.savefig(save_path / "Figures" / "KL-Divergence")
plt.close(fig)


scores = np.array(
    [
        nabqra.scoring.continous_kl_divergence(dist1.pdf, dist2.pdf, (-3, 3), limit=200)[0],
        nabqra.scoring.continous_kl_divergence(dist2.pdf, dist1.pdf, (-3, 3), limit=200)[0],
    ]
)
kl_table = pd.DataFrame(np.atleast_2d(scores), columns=["$KL(F||G)$", "$KL(G||F)$"])
kl_table.style.format(precision=2).hide(axis="index").to_latex(
    save_path / "Tables" / "Kullback-Leibler.tex",
    hrules=True,
    clines="skip-last;data",
    convert_css=True,
    position="ht",
    position_float="centering",
    multicol_align="r",
    multirow_align="r",
    label=r"tab:evaluation:kl",
    caption=(
        "Kullback-Leibler divergence of the two example distributions.",
        "Kullback-Leibler divergence",
    ),
)

# %% MAE

X = np.linspace(-3, 3, num=1000)
X2 = np.array([1, 10, 20])

Ys = np.add.outer(np.abs(X), X2) / 2

fig, ax = plt.subplots()

ax.plot(X, Ys)

ax.legend([f"$y_2 = {x2}$" for x2 in X2])
ax.set_xlabel("$y_1$")
ax.set_ylabel("MAE")

fig.savefig(save_path / "Figures" / "MAE")
plt.close(fig)

# %% RMSE

X = np.linspace(-3, 3, num=1000)
X2 = np.array([1, 10, 100])

Ys = np.sqrt(np.add.outer(X**2, X2**2) / 2)
Y_max = Ys.max(axis=0)

labels = [f"$y_2 = {x2}$" for x2 in X2]

fig, ax = nabqra.plotting.multi_y_plot(X, Ys.T, labels=labels, ylims=list(zip(Y_max - 1.8, Y_max)))

ax.get_legend().set(loc="lower left")
ax.set_xlabel("$y_1$")
ax.set_title("RMSE")

fig.savefig(save_path / "Figures" / "RMSE")
plt.close(fig)

# %% QS


def quantile_loss(q, y, f):
    e = y - f
    return np.maximum(q * e, (q - 1) * e)


quantiles = [0.001, 0.5, 0.95]

X = np.linspace(-4, 4, num=1000)
Y = np.zeros((len(quantiles), len(X)))

fig, ax = plt.subplots()

for q in quantiles:
    loss = quantile_loss(q, X, stats.norm.ppf(q))
    l = ax.plot(X, loss, label=f"q = {q}")
    # ax.vlines(stats.norm.ppf(q), -0.1, 2.5, color = l[0].get_color(), linestyle = "--")

ax.set_xlim([-4, 4])
ax.set_ylim([-0.2, 2])
ax.set_xlabel("$y$")
ax.set_ylabel("$QS(y)$")
ax.legend()

fig.savefig(save_path / "Figures" / "QS")
plt.close(fig)


# %% CRPS

dist = stats.norm()
example_point = -0.5
X = np.linspace(-3, 3, 1000)
sigmas = [0.1, 1, 10]
Ys = np.array([ps.crps_gaussian(X, 0, sigma) for sigma in sigmas]).T


fig, ax = plt.subplots()

ax.plot(X, dist.cdf(X), color="black", label=r"True distribution")
ax.plot(X, np.heaviside(X - example_point, 1), label="Observed Distribution")
ax.vlines(example_point, 0, 1, color="black", linestyle="--", label=r"Observed x")
ax.fill_between(X, dist.cdf(X), np.heaviside(X - example_point, 1), alpha=0.3)

ax.set_xlabel("$x$")
ax.set_ylabel("$F(x)$")
ax.legend()

fig.savefig(save_path / "Figures" / "CRPS observation")
plt.close(fig)


fig, ax = plt.subplots()
ax.plot(X, Ys)
ax.legend([f"$N(0,{sigma}^2)$" for sigma in sigmas])
ax.set_ylabel("$CRPS$")
ax.set_xlabel("$x$")

fig.savefig(save_path / "Figures" / "CRPS")
plt.close(fig)

# %% Vars

process = pd.read_pickle(load_path / "process.pkl")
scores = pd.read_pickle(load_path / "scores.pkl")
comp = np.load(load_path / "weight comp.npz")

# %%
fig, ax = plt.subplots()

nabqra.plotting.band_plot(np.arange(24), *process.loc["AR(2) model"].values.T, ax=ax, label="AR(2) model")
nabqra.plotting.band_plot(np.arange(24), *process.loc["AR(1) model"].values.T, ax=ax, label="AR(1) model")
ax.set_xlabel("t")
ax.set_ylabel("$y_t$")
ax.legend()

fig.savefig(save_path / "Figures" / "Process")
plt.close(fig)


e_correct = np.ma.masked_equal(comp["e_correct"], 0.0, copy=False)
e_wrong = np.ma.masked_equal(comp["e_wrong"], 0.0, copy=False)

vmin = min(e_wrong.min(), e_correct.min())
vmax = min(e_wrong.max(), e_correct.max())
fig, axes = plt.subplots(1, 2, layout="constrained", sharex=True, sharey=True)
im = axes[0].imshow(comp["e_correct"], vmin=vmin)
axes[1].imshow(comp["e_wrong"], vmax=vmax)
axes[0].set_title("AR(2) model")
axes[1].set_title("AR(1) model")

fig.colorbar(im, ax=axes)
fig.savefig(save_path / "Figures" / "Variogram")
plt.close(fig)


fig, ax = plt.subplots()
sns.kdeplot(
    (scores).melt(var_name="Order"),
    x="value",
    hue="Order",
    fill=True,
    common_grid=False,
    common_norm=False,
    ax=ax,
)
ax.set_xlabel("VarS")
ax.set_xlim([0, 5])
fig.savefig(save_path / "Figures" / "Vars dist")
plt.close(fig)


w1 = np.ma.masked_equal(comp["weight"], 0.0, copy=False)

fig, ax = plt.subplots(1)
im = ax.imshow(w1 / w1.max(), vmin=0.01, vmax=1, norm="log")
fig.colorbar(im)
fig.savefig(save_path / "Figures" / "Weight")
plt.close(fig)

s1 = np.ma.masked_equal(comp["diffs"].mean(axis=0), 0.0, copy=False)
s2 = np.ma.masked_equal(comp["diffs_weight"].mean(axis=0), 0.0, copy=False)


vmin = min(s1.min(), s2.min())
vmax = max(s1.max(), s2.max())

fig, axes = plt.subplots(1, 2, layout="constrained", sharex=True, sharey=True)
im = axes[0].imshow(s1 / s1.max(), vmin=0.01, vmax=1, norm="log")
axes[1].imshow(s2 / s2.max(), vmin=0.01, vmax=1, norm="log")
fig.colorbar(im, ax=axes)
fig.savefig(save_path / "Figures" / "score diff")
plt.close(fig)

diff_scores = pd.DataFrame({"Equal": comp["scores"], "Inverse Distance": comp["scores_weight"]}).melt(
    var_name="Weight", value_name="Score Difference"
)


fig, ax = plt.subplots()
sns.kdeplot(diff_scores, x="Score Difference", hue="Weight", fill=True, ax=ax)
fig.savefig(save_path / "Figures" / "diff dist")
plt.close(fig)

diff_scores.set_index("Weight", append=True).groupby("Weight").mean()

(diff_scores.set_index("Weight", append=True) > 0).groupby(level=1).mean().T.style.format("{:.2f}").to_latex(
    save_path / "Tables" / "acceptance rate.tex",
    hrules=True,
    clines="skip-last;data",
    convert_css=True,
    position="htb",
    position_float="centering",
    multicol_align="c",
    multirow_align="r",
    label="tab:evaluation:precision",
    caption=(
        "Precision of the \gls{vars}, for the equally weighted score and the score weighted with inverse distance",
        "Precision by Weight",
    ),
)

# %% Expected score as a function of variance

sigmas = np.logspace(-1, 3, 100)
N = 10000
K = 10
z = stats.norm(scale=sigmas).rvs((K, N, len(sigmas)), random_state=rng)

mae = np.mean(np.abs(z), axis=0)
rmse = np.sqrt(np.mean(z**2, axis=0))
crps = np.mean([ps.crps_gaussian(z[:, :, i], 0, s) for i, s in enumerate(sigmas)], axis=1).T

VarS_mean = np.mean(np.abs(z) ** 0.5, axis=(0, 1))
VarS = np.mean(np.abs(np.abs(z) ** 0.5 - VarS_mean) ** 2, axis=0)

fig, ax = plt.subplots()

ax.plot(sigmas, np.std(mae, axis=0))
ax.plot(sigmas, np.std(rmse, axis=0))
ax.plot(sigmas, np.std(crps, axis=0))
ax.plot(sigmas, np.std(VarS, axis=0))


plt.legend(["MAE", "RMSE", "CRPS", "VarS"])
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Expected Score")
ax.set_xlabel(r"$\sigma$")

fig.savefig(save_path / "Figures" / "Score variance")
plt.close(fig)
