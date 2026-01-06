# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from pathlib import Path

PATH = Path("C:/Users/kpfs/Projects/Correlation-structure-article")

zones = ["DK1-onshore", "DK2-onshore", "DK1-offshore", "DK2-offshore"]
example_date = pd.to_datetime(["2023-07-01", "2023-07-14"], utc=True)

# %% raw data
ens = pd.read_pickle(PATH / "Data" / "ensembles.pkl")
obs = pd.read_pickle(PATH / "Data" / "observations.pkl")


fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), squeeze=True)
axes = axes.ravel()
for i, zone in enumerate(zones):

    tmp_ens = ens.loc[zone].loc[example_date[0] : example_date[1]]
    tmp_obs = obs.loc[zone].loc[example_date[0] : example_date[1]]
    axes[i].plot(
        tmp_ens.index,
        tmp_ens,
        color="black",
        alpha=0.05,
        label="Ensemble",
    )
    axes[i].scatter(tmp_obs.index, tmp_obs, s=1, label="Actual Production")

    axes[i].set_title(zone)
    axes[i].tick_params(axis="x", labelrotation=20)

axes[2].set_xlim(*example_date)
fig.supxlabel("Date")
fig.supylabel("MWh")
fig.set_layout_engine("constrained")

fig.savefig(PATH / "Results" / "Graphs" / "data_example.pdf")

# %% data summary

tmp = obs.unstack(level=0).describe().drop("count")
tmp.style.format(precision=2).to_latex(
    PATH / "Results" / "Tables" / "observations_summary.tex",
    position="htb",
    hrules=True,
)

tmp = (
    ens.unstack(level=0)
    .stack(level=0, future_stack=True)
    .describe()
    .drop("count")
)
tmp.style.format(precision=2).to_latex(
    PATH / "Results" / "Tables" / "ensembles_summary.tex",
    position="htb",
    hrules=True,
)

# %% training loss

train = pd.read_pickle(PATH / "Data" / "modeltraining.pkl")

fig, axes = plt.subplots(2, 2, figsize=(8, 6), squeeze=True)
axes = axes.ravel()
for i, zone in enumerate(zones):
    for model, df in train.loc[zone].groupby(level=0):
        l = axes[i].plot(df["Train loss"].values, label=model)
        axes[i].plot(
            df["Validation loss"].values, ls="--", color=l[0].get_color()
        )
    axes[i].set_title(zone)
    axes[i].set_ylim(
        [
            np.nanmin(train.loc[zone].values),
            train.loc[zone].xs(25, level=1).values.max(),
        ]
    )


axes[1].plot(0, 0, color="black", label="Train Loss")
axes[1].plot(0, 0, color="black", ls="--", label="Validation Loss")
axes[1].legend()

fig.supxlabel("Epoch")
fig.supylabel("Quantile Loss")
fig.set_layout_engine("constrained")

fig.savefig(PATH / "Results" / "Graphs" / "loss_curves.pdf")

# %% Marginal models

marginal_quantiles = pd.read_pickle(PATH / "Data" / "marginal_quantiles.pkl")

fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), squeeze=True)
axes = axes.ravel()
for i, zone in enumerate(zones):
    q = marginal_quantiles.loc[zone, "Latent"]
    x = q.index

    l = axes[i].plot(x, q["0.50"])

    for j in range(1, len(marginal_quantiles.columns) // 2):
        axes[i].fill_between(
            x,
            q.iloc[:, j],
            q.iloc[:, -j - 1],
            color=l[0].get_color(),
            alpha=0.2,
        )

    # Observations
    axes[i].scatter(obs.loc[zone].index, obs.loc[zone], s=3, color="black")

    axes[i].set_title(zone)
    axes[i].tick_params(axis="x", labelrotation=20)

axes[2].set_xlim(*example_date)
fig.supxlabel("Date")
fig.supylabel("MWh")
fig.set_layout_engine("constrained")

fig.savefig(PATH / "Results" / "Graphs" / "marginal_example.pdf")


# %% pseudo residual

marginal_quantiles = pd.read_pickle(PATH / "Data" / "marginal_quantiles.pkl")
pseudo = pd.read_pickle(PATH / "Data" / "pseudoresidual.pkl")

fig, ax = plt.subplots(figsize=(8, 6))

pr = pseudo.loc["DK1-onshore", "Latent"]
x = pr.index

l = ax.axhline(0, label="Median")

for i in range(1, len(marginal_quantiles.columns) // 2):
    q1 = stats.norm.ppf(float(marginal_quantiles.columns[i]))
    q2 = stats.norm.ppf(float(marginal_quantiles.columns[-i - 1]))
    ax.axhspan(q1, q2, color=l.get_color(), alpha=0.2)

# Observations
ax.scatter(pr.index, pr, s=3, color="black", label="Actual production")

ax.set_title(zone)
ax.set_xlabel("Date")
ax.set_ylabel("MWh")
ax.tick_params(axis="x", labelrotation=20)
ax.set_xlim(*example_date)

fig.set_layout_engine("constrained")
fig.savefig(PATH / "Results" / "Graphs" / "pseudoresidual.pdf")

# %% arima params

params = pd.DataFrame(
    columns=pd.MultiIndex.from_product([zones, ["Latent", "Simple"]]),
    index=["ar1", "ma1", "AR1", "sigma"],
    dtype=np.float64,
)
for f in (PATH / "Models" / "sarima").glob("*"):

    zone, model = f.stem.split("_")
    data = np.load(f)

    params[zone, model] = data

params.style.format(precision=2).to_latex(
    PATH / "Results" / "Tables" / "parameters.tex",
    hrules=True,
    clines="all;data",
)
# %% Autocorrelation models

sims = pd.read_pickle(PATH / "Data" / "simulation.pkl")
sim_quants = pd.DataFrame(
    np.quantile(sims, np.arange(0, 1.01, 0.05), axis=1).T,
    index=sims.index,
    columns=[f"{x:.2f}" for x in np.arange(0, 1.01, 0.05)],
)

fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), squeeze=True)
axes = axes.ravel()
for i, zone in enumerate(zones):
    q = sim_quants.loc[zone, "Latent"]
    x = q.index

    l = axes[i].plot(x, q["0.50"])

    for j in range(1, len(marginal_quantiles.columns) // 2):
        axes[i].fill_between(
            x,
            q.iloc[:, j],
            q.iloc[:, -j - 1],
            color=l[0].get_color(),
            alpha=0.2,
        )

    # Observations
    axes[i].scatter(obs.loc[zone].index, obs.loc[zone], s=3, color="black")

    axes[i].set_title(zone)
    axes[i].tick_params(axis="x", labelrotation=20)

axes[2].set_xlim(*example_date)
fig.supxlabel("Date")
fig.supylabel("MWh")
fig.set_layout_engine("constrained")

fig.savefig(PATH / "Results" / "Graphs" / "corrolation_example.pdf")


# %% Scores

scores = pd.read_pickle(PATH / "Data" / "scores.pkl")
scores.style.format(precision=2).to_latex(
    PATH / "Results" / "Tables" / "scores.tex", hrules=True, clines="all;data"
)

scores_percent = pd.read_pickle(PATH / "Data" / "scores_percent.pkl")
scores_percent.style.format(precision=2).to_latex(
    PATH / "Results" / "Tables" / "scores_percent.tex",
    hrules=True,
    clines="all;data",
)
