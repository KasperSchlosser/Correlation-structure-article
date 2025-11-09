import pathlib
import tomllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nabqra.plotting as nplt

from pandas import IndexSlice as idx

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Basis"
save_path = PATH / "Results" / "Basis"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]

scores = pd.read_pickle(load_path / "Basis scores.pkl").astype(np.float64)
history = pd.read_pickle(load_path / "History.pkl")
preds = pd.read_pickle(load_path / "Basis Quantiles.pkl").sort_index()
resids = pd.read_pickle(load_path / "Basis Residuals.pkl").astype(np.float64)
observations = pd.read_pickle(load_path / ".." / "Data" / "cleaned_observations.pkl")

models = preds.index.unique(1)

# %% NN training


fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(14, 8))
axes = axes.ravel()
axes[-1].plot(1, 1, color="none", label="Model:")
for ax, zone in zip(axes, zones):

    tmp = history.loc[zone, ["loss", "val_loss"]].unstack("Model").loc[1:]

    for m in tmp.columns.unique(1):
        l = ax.semilogx(tmp["loss", m], label=m)
        ax.semilogx(tmp["val_loss", m], linestyle="--", color=l[0].get_color())

    # ax.legend()
    ax.set_title(zone)
    ax.set_ylim((tmp.min(axis=None) * 0.95, tmp.min(axis=1)[1] * 1.05))


axes[-1].plot(1, 1, color="none", label=" ")
axes[-1].plot(1, 1, color="none", label="Type:")
axes[-1].plot(1, 1, color="black", linestyle="-", label="Train loss")
axes[-1].plot(1, 1, color="black", linestyle="--", label="Validation loss")


fig.legend(*ax.get_legend_handles_labels())


fig.supxlabel("Epoch")
fig.supylabel("Loss")

fig.savefig(save_path / "Figures" / "Loss")
plt.close(fig)

# %% Model examples

zone = "DK1-onshore"
obs = observations.loc[zone]
# period = [np.datetime64("2023-10-17"), np.datetime64("2023-11-02")]
period = [np.datetime64("2024-01-19"), np.datetime64("2024-01-25")]

for model in models:

    df = preds.loc[zone, model]

    fig, ax = plt.subplots()

    nplt.band_plot(df.index, df["0.50"], df["0.05"], df["0.95"], ax=ax)
    ax.scatter(obs.index, obs, color="black", s=1)
    ax.set_xlim(period)
    # ax.set_ylim([-10, 3800])
    ax.tick_params(axis="x", labelrotation=10)
    ax.set_ylabel("Production (MWh)")
    # ax.set_title(model)

    fig.savefig(save_path / "Figures" / f"{model} example")
    plt.close(fig)

# compare feature and ensemble

fig, ax = plt.subplots()

nplt.band_plot(
    observations.loc[zone].index,
    *preds.loc[idx[zone, "Feature"], ["0.50", "0.05", "0.95"]].values.T,
    ax=ax,
    alpha=0.2,
    label="Feature",
)
nplt.band_plot(
    observations.loc[zone].index,
    *preds.loc[idx[zone, "Ensemble"], ["0.50", "0.05", "0.95"]].values.T,
    ax=ax,
    alpha=0.2,
    label="Ensemble",
)


ax.scatter(observations.loc[zone].index, observations.loc[zone], color="black", label="Observation")
ax.set_xlim(period)
ax.legend()
ax.set_ylabel("Production(MWh)")
ax.tick_params(axis="x", labelrotation=10)
fig.savefig(save_path / "Figures" / "Comparison")
plt.close(fig)

# %% pseudo residuals


for zone, df in resids.groupby(level=0):
    print(zone)
    figs, _ = nplt.diagnostic_plots(
        df.values, df.index.get_level_values(1), save_path / "Figures" / "Residuals" / f"{zone}", closefig=True
    )


# %% scores

# geometric mean
avg_score = np.exp(np.log(scores).groupby(level=1).mean())
avg_score = pd.concat([avg_score], keys=["Geometric mean score"])
scores = pd.concat([scores, avg_score])

ratios = scores.div(scores.groupby(level=0).min(), axis=0, level=0)
form = np.full(ratios.shape, "", dtype=object)

form[ratios == 1] += "font-weight: bold; font-style: italic; "
form[ratios <= 1.1] += "background-color: #269BE3; "
# form[ratios > 1.1] += "background-color: #FF5C61; "

form = pd.DataFrame(form, index=scores.index, columns=scores.columns)


(
    scores.style.format(precision=2, na_rep="")
    .apply(
        lambda x: form,
        axis=None,
    )
    .to_latex(
        save_path / "Tables" / "Basis scores.tex",
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        position="htb",
        position_float="centering",
        multicol_align="c",
        multirow_align="r",
        label="tab:basis:scores",
        caption=(
            "Scores for the estimated marginal distributions. "
            "Bold score indicated the minimum scores for the the zone and measure. "
            "Blue scores are within 10\% of the minimum score",
            "Marginal distribution scores.",
        ),
    )
)
