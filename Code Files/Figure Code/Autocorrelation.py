import pathlib
import tomllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nabqra.plotting as nplt
import scipy.stats as stats

from pandas import IndexSlice as idx

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Autocorrelation"
save_path = PATH / "Results" / "Autocorrelation"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]

# scores = pd.read_pickle(load_path / "Forecast scores.pkl")
# preds = pd.read_pickle(load_path / "Forecast.pkl").astype(np.float64)
# resids_onestep = pd.read_pickle(load_path / "Residuals onestep.pkl").astype(np.float64)
# observations = pd.read_pickle(load_path / ".." / "Data" / "cleaned_observations.pkl")
# params = pd.read_pickle(load_path / "Model Params.pkl")

# models = preds.index.unique(1)

selection = pd.read_pickle(load_path / "Model selection.pkl")
sarma = pd.read_pickle(load_path / "Sarma forecasts.pkl")

residuals = pd.read_pickle(load_path / "Sarma residuals.pkl")
params = pd.read_pickle(load_path / "Params.pkl")


observations = pd.read_pickle(load_path / ".." / "Data" / "cleaned_observations.pkl")
feature = pd.read_pickle(load_path / ".." / "Basis" / "Basis quantiles.pkl").xs("Feature", level=1)
ensemble = pd.read_pickle(load_path / ".." / "Basis" / "Basis quantiles.pkl").xs("Ensemble", level=1)

# %%compare feature and sarma

zone = "DK1-onshore"
obs = observations.loc[zone]
period = [np.datetime64("2024-01-19"), np.datetime64("2024-01-25")]

for zone in zones:
    fig, ax = plt.subplots()
    index = sarma.loc[zone].index

    # nplt.band_plot(
    #     index, *ensemble.loc[idx[zone, index], ["0.50", "0.05", "0.95"]].values.T, ax=ax, alpha=0.2, label="Ensemble"
    # )
    nplt.band_plot(
        index, *feature.loc[idx[zone, index], ["0.50", "0.05", "0.95"]].values.T, ax=ax, alpha=0.2, label="Feature"
    )
    nplt.band_plot(index, *sarma.loc[zone].values.T, ax=ax, alpha=0.2, label="SARMA")

    ax.scatter(index, observations.loc[zone, index], color="black", label="Observation")
    ax.set_xlim(period)
    ax.legend()
    ax.set_ylabel("Production(MWh)")
    ax.tick_params(axis="x", labelrotation=10)

    fig.savefig(save_path / "Figures" / f"{zone} example")
    plt.close(fig)

# %% Residual

for zone, df in residuals.groupby(level=0):
    print(zone)
    figs, _ = nplt.diagnostic_plots(
        df.values.squeeze(),
        df.index.get_level_values(1),
        save_path=save_path / "Figures" / "Residuals" / f"{zone}",
        closefig=True,
    )

# %% variograms

variograms = np.load(load_path / "variograms.npz")

for zone in zones:
    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(variograms[f"{zone} expected variogram"], aspect="equal")
    fig.colorbar(im, ax=ax, label="$E(|y_i - y_j|^{0.5})$", pad=0.01)
    ax.set_xlabel("i")
    ax.set_ylabel("j")

    fig.savefig(save_path / "Figures" / f"{zone} variogram")
    plt.close(fig)


# %% param tabel
tmp = params.astype(np.float64)
sigmas = tmp.loc["sigma2"]
tmp = tmp.drop("sigma2").round(3)

tmp.index = pd.MultiIndex.from_tuples(tmp.index.to_series().str.split(".L"))
tmp.index = tmp.index.set_levels(tmp.index.levels[1].astype(int), level=1)
tmp = tmp.reset_index(names=["Parameter", "Lag"])
tmp = tmp.set_index("Parameter")

tmp.loc[["ar.S", "ma.S"], "Lag"] = (tmp.loc[["ar.S", "ma.S"], "Lag"] // 24).values

tmp = tmp.set_index("Lag", append=True)
tmp = "$" + tmp.xs("coef", axis=1, level=1).astype(str) + " \pm " + tmp.xs("std err", axis=1, level=1).astype(str) + "$"
tmp = tmp.map(lambda x: "-" if "nan" in x else x)
tmp = tmp.T.stack("Lag", future_stack=True)
tmp = tmp.rename_axis([None, None], axis=0)
tmp = tmp.rename_axis(None, axis=1)
tmp.columns = [r"$\phi_i$", r"$\theta_i$", r"$\Phi_{24, i}$", r"$\Theta_{24,i}$"]
tmp = tmp.stack().unstack(level=0).swaplevel(0, 1, axis=0).sort_index(level=[0, 1], ascending=[False, True])


sigmas = sigmas.unstack(level=1).round(3)
sigmas = "$" + sigmas["coef"].astype(str) + " \pm " + sigmas["std err"].astype(str) + "$"

tmp.loc[("$\sigma^2$", ""), :] = sigmas

tmp = tmp.sort_index(level=0, key=lambda x: [{"s": 1, "p": 2, "t": 3, "P": 4, "T": 5}[k[2]] for k in x])
# tmp.columns = pd.MultiIndex.from_tuples([(x, sigmas[x]) for x in tmp.columns], names = ["Zone", "$\sigma$"])

tmp.style.format(precision=2, na_rep="").to_latex(
    save_path / "Tables" / "Parameters.tex",
    hrules=True,
    clines="skip-last;data",
    convert_css=True,
    position="htb",
    position_float="centering",
    multicol_align="c",
    multirow_align="c",
    column_format="lccccc",
    label="tab:autocorrelation:parameters",
    caption=("Estimated parameters for the SARMA model. ", "Estimated parameters."),
)


# %% Result table

sarma_scores = pd.read_pickle(load_path / "Sarma scores.pkl")
sarma_scores = pd.concat([sarma_scores], keys=["SARMA"], names=["Model"]).swaplevel(0, 1)
nodata_scores = pd.read_pickle(load_path / "Sarma nodata scores.pkl")
nodata_scores = pd.concat([nodata_scores], keys=["SARMA - no obs"], names=["Model"]).swaplevel(0, 1)
basis_scores = pd.read_pickle(load_path / ".." / "Basis" / "Basis scores.pkl").loc[
    idx[:, ["Ensemble", "Ensemble - Raw", "Feature"]], :
]
scores = pd.concat((basis_scores, sarma_scores, nodata_scores), axis=0).astype(np.float64)

avg_score = np.exp(np.log(scores).groupby(level=1).mean())
avg_score = pd.concat([avg_score], keys=["Geometric mean score"])


scores = pd.concat([scores, avg_score]).sort_index(level=[0, 1]).sort_index(level=[0, 1])

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
        save_path / "Tables" / "Scores.tex",
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        position="htb",
        position_float="centering",
        multicol_align="c",
        multirow_align="r",
        label="tab:autocorrelation:scores",
        caption=(
            "Scores for the estimated marginal distributions. "
            "Bold scores indicated the minimum scores for the the zone and measure. "
            "Scores colored blue are within 10\% of the minimum score",
            "Marginal distribution scores.",
        ),
    )
)
