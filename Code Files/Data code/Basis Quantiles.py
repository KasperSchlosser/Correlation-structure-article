import pathlib
import tomllib

import pandas as pd
import numpy as np

from nabqra.taqr import run_taqr
from nabqra.misc import fix_quantiles
from nabqra.quantiles import spline_model
from nabqra.scoring import calc_scores
from pandas import IndexSlice as idx


PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Basis"
save_path = PATH / "Data" / "Basis"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_lims = parameters["Zone-Limits"]
    quantiles = np.array(parameters["Quantiles"])
    quantiles_str = [f"{q:.2f}" for q in quantiles]

obs = pd.read_pickle(load_path / ".." / "Data" / "cleaned_observations.pkl")
ensembles = pd.read_pickle(load_path / ".." / "Data" / "cleaned_ensembles.pkl")

index = obs.index.unique(1)
train_index = index[: parameters["train_size"]]
test_index = index[parameters["train_size"] :]

features = pd.read_pickle(load_path / "Features.pkl")
features.loc[idx[:, :], "const"] = 1

quantile_res = pd.read_pickle(load_path / "NN quantiles.pkl")

rng = np.random.default_rng(42)
n_sim = 1000


# %% add ensemble quantiles
ensembles_quant = ensembles.quantile(quantiles, axis=1).T
ensembles_quant.columns = quantiles_str
ensembles_quant = pd.concat([ensembles_quant], keys=["Ensemble"], names=["Model"]).swaplevel(0, 1)
quantile_res = pd.concat((quantile_res, ensembles_quant)).sort_index()

# %% taqr new features

taqr_quant = pd.DataFrame(
    index=pd.MultiIndex.from_product((zones, index), names=["Zone", "Time"]),
    columns=quantile_res.columns,
    dtype=np.float64,
)

init = 500
for zone in zones:

    print(f"feature-taqr {zone}")

    tmp = run_taqr(
        features.loc[zone].iloc[-10000:],
        obs.loc[zone].iloc[-10000:],
        quantiles,
        init,
        len(obs.loc[zone].iloc[-10000:]),
        init,
    )
    taqr_quant.loc[idx[zone, index[-10000 + init + 2 :]], :] = np.stack(tmp[0], axis=1)

taqr_quant = pd.concat([taqr_quant], keys=["TAQR"], names=["Model"]).swaplevel(0, 1)
quantile_res = pd.concat((quantile_res, taqr_quant))

# %% calculate scores

scores = pd.DataFrame(
    index=quantile_res.droplevel(2).index.unique(), columns=["MAE", "RMSE", "CRPS", "VarS"], dtype=np.float64
)

for zone, model in scores.index:

    print(zone, model)
    # fix quantiles if bad

    est_quant = fix_quantiles(quantile_res.loc[zone, model, test_index], *zone_lims[zone])
    dist = spline_model(quantiles, *zone_lims[zone])

    sim = dist.make_sim(est_quant.values, n_sim=n_sim, random_state=rng)

    print("sim done")

    scores.loc[zone, model, :] = calc_scores(obs.loc[zone, test_index].values, est_quant["0.50"].values, sim)


# %% add raw ensemble scores

ensemble_scores = pd.DataFrame(columns=scores.columns, index=zones)
for zone in zones:
    ens = ensembles.loc[idx[zone, test_index], :]

    tmp = calc_scores(obs.loc[zone, test_index].values, ens["HPE"].values, ens.values)
    ensemble_scores.loc[zone, :] = tmp

ensemble_scores = pd.concat([ensemble_scores], keys=["Ensemble - Raw"], names=["Model"]).swaplevel(0, 1)
scores = pd.concat((scores, ensemble_scores)).sort_index()


# %% find pseudo Residuals

residuals = pd.Series(index=obs.index, name="Feature", dtype=np.float64)

for zone in zones:

    print(zone)
    est_quant = fix_quantiles(quantile_res.loc[zone, "Feature"], *zone_lims[zone])
    dist = spline_model(quantiles, *zone_lims[zone])

    residuals.loc[zone] = dist.transform(est_quant.values, obs.loc[zone].values)[0].squeeze()

# %% save

quantile_res.to_csv(save_path / "Basis quantiles.csv")
quantile_res.to_pickle(save_path / "Basis quantiles.pkl")

scores.to_csv(save_path / "Basis scores.csv")
scores.to_pickle(save_path / "Basis scores.pkl")

residuals.to_csv(save_path / "Basis residuals.csv")
residuals.to_pickle(save_path / "Basis residuals.pkl")
