import pathlib
import tomllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nabqra
import scipy.stats as stats

from pandas import IndexSlice as idx

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Distribution"
save_path = PATH / "Results" / "Distribution"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]

rng = np.random.default_rng(42)

# %% load data

estimate_df = pd.read_pickle(load_path / "estimates.pkl")
score_df = pd.read_pickle(load_path / "scores.pkl").T
tail_df = pd.read_pickle(load_path / "tail_problem.pkl")

dists = estimate_df.columns.unique(1)
models = estimate_df.columns.unique(0)

# %% examples

lims = [-3, 3]


# dist = stats.norm()
dist = stats.dweibull(2)
X = np.linspace(*lims, 1000)
X_obs = [-1, -0.3, -0.2, -0.1, 0, 0.3, 2]
Q_obs = dist.cdf(X_obs)

fig, ax = plt.subplots()

ax.plot(X, dist.cdf(X), color="black", label="True distribution", linestyle="--")

for model in [
    nabqra.quantiles.linear_model(Q_obs, *lims, tail_correction=False),
    nabqra.quantiles.linear_model(Q_obs, *lims, tail_correction=True),
    nabqra.quantiles.spline_model(Q_obs, *lims),
]:
    model.fit(X_obs)

    ax.plot(X, model.cdf(X))
ax.set_xlabel("x")
ax.set_ylabel("F(x)")
ax.legend(["True", "Linear", "Linear Quadratic tail", "Spline"])

fig.savefig(save_path / "Figures" / "Example cdf fit")
plt.close(fig)

fig, ax = plt.subplots()

ax.plot(X, dist.pdf(X), color="black", label="True distribution", linestyle="--")

for model in [
    nabqra.quantiles.linear_model(Q_obs, *lims, tail_correction=False),
    nabqra.quantiles.linear_model(Q_obs, *lims, tail_correction=True),
    nabqra.quantiles.spline_model(Q_obs, *lims),
]:
    model.fit(X_obs)

    ax.plot(X, model.pdf(X))
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(["True", "Linear", "Linear Quadratic tail", "Spline"])

fig.savefig(save_path / "Figures" / "Example pdf fit")
plt.close(fig)

# example data
ex_data = pd.DataFrame({"$q_i$": X_obs, "$p_i$": Q_obs}, index=pd.Series(np.arange(1, len(Q_obs) + 1), name="$i$"))
ex_data.T.style.format(precision=3).to_latex(
    save_path / "Tables" / "Example data.tex",
    hrules=True,
    clines="skip-last;data",
    convert_css=True,
    position="htb",
    position_float="centering",
    multicol_align="r",
    multirow_align="r",
    label="tab:distributions:exampledata",
    caption=(
        "Values and quantiles for illustrative example",
        "Example Data points",
    ),
)

# %% dist plots
X = estimate_df.index
fig, ax = plt.subplots()

ax.plot(estimate_df.index, estimate_df.loc[:, idx["True", :, "cdf"]])
ax.legend(dists)
ax.set_xlabel("$x$")
ax.set_ylabel("$F(x)$")

fig.savefig(save_path / "Figures" / "Comparison cdf")
plt.close(fig)

fig, ax = plt.subplots()

ax.plot(estimate_df.index, estimate_df.loc[:, idx["True", :, "pdf"]])
ax.legend(dists)
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")

fig.savefig(save_path / "Figures" / "Comparison pdf")
plt.close(fig)

for dist in dists:

    fig, ax = plt.subplots()

    ax.plot(X, estimate_df["True", dist, "cdf"], color="black")
    ax.plot(X, estimate_df.xs((dist, "cdf"), axis=1, level=(1, 2)).iloc[:, 1:], alpha=0.9)
    ax.legend(models)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$F(x)$")

    fig.savefig(save_path / "Figures" / (dist + " cdf"))
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.plot(X, estimate_df["True", dist, "pdf"], color="black")
    ax.plot(X, estimate_df.xs((dist, "pdf"), axis=1, level=(1, 2)).iloc[:, 1:], alpha=0.9)
    ax.legend(models)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")

    fig.savefig(save_path / "Figures" / (dist + " pdf"))
    plt.close(fig)

# %% test quantiles

test_quantiles = pd.read_pickle(load_path / "test quantiles.pkl").sort_index().T
test_quantiles.columns = [f"$F^{{-1}}({x:.2f})$" for i, x in enumerate(test_quantiles.columns)]

test_quantiles.style.format(precision=2).to_latex(
    save_path / "Tables" / "test quantiles.tex",
    hrules=True,
    clines="skip-last;data",
    label="tab:distributions:testinterpolation",
    convert_css=True,
    position="htb",
    position_float="centering",
    multicol_align="r",
    multirow_align="r",
    caption=(
        "The interpoilation points used for testing the interpolation methods",
        "Test interpolation points",
    ),
)


# %% score table

score_df.style.format(precision=2).highlight_min(axis=1, props="font-weight:bold;").to_latex(
    save_path / "Tables" / "scores.tex",
    hrules=True,
    clines="skip-last;data",
    convert_css=True,
    position="htb",
    position_float="centering",
    label="tab:distributions:testres",
    multicol_align="r",
    multirow_align="r",
    caption=(
        "The Wasserstein distance (with \( p = 1 \) and \( p = 2 \)) and \gls{kl} for the three interpolation methods.",
        "Interpolation Scores",
    ),
)

# %% pseudo residuals

dist1 = stats.norm()
dist2 = stats.t(5, loc=0.5)
K = 1000

res = dist1.rvs(K, random_state=rng)
cdf_res = dist1.cdf(res)

nabqra.plotting.diagnostic_plots(
    dist1.ppf(cdf_res),
    np.arange(K),
    save_path=save_path / "Figures" / "Residuals" / "normal distribution",
    closefig=True,
)
nabqra.plotting.diagnostic_plots(
    dist2.ppf(cdf_res),
    np.arange(K),
    save_path=save_path / "Figures" / "Residuals" / "t distribution",
    closefig=True,
)

# %% tail

fig, ax = nabqra.plotting.band_plot(np.arange(len(tail_df)), *tail_df["True"].values.T, band_label="True interval")

for name in list(models)[1:]:
    l = ax.plot(tail_df[name, "Mean"], label=name)
    ax.plot(tail_df[name, "Lower"], color=l[0].get_color(), linestyle="--")
    ax.plot(tail_df[name, "Upper"], color=l[0].get_color(), linestyle="--")

ax.legend()
ax.set_xlabel("t")

fig.savefig(save_path / "Figures" / "Tail Problem")
plt.close(fig)
