import pathlib
import tomllib

import pandas as pd
import numpy as np
import nabqra
import statsmodels.api as sm

from nabqra.scoring import calc_scores


PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Basis"
save_path = PATH / "Data" / "Autocorrelation"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_lims = parameters["Zone-Limits"]
    quantiles = np.array(parameters["Quantiles"])
    quantiles_str = [f"{q:.2f}" for q in quantiles]

basis_quantiles = pd.read_pickle(load_path / "Basis quantiles.pkl")
obs = pd.read_pickle(load_path / ".." / "Data" / "cleaned_observations.pkl")
pseudores = pd.read_pickle(load_path / "Basis residuals.pkl")

index = obs.index.unique(1)
train_index = index[: parameters["train_size"]]
test_index = index[parameters["train_size"] :]

rng = np.random.RandomState(42)


def params_to_pd(res):
    data = np.array(res.summary().tables[1].data)
    return pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])[["coef", "std err"]]


# %%
# selection = pd.DataFrame(
#     index=pd.MultiIndex.from_product(
#         [zones, range(4), range(4), range(3), range(3)], names=["Zone", "p", "q", "P", "Q"]
#     ),
#     columns=["AIC", "BIC"],
#     dtype=np.float64(),
# )

selection = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [zones, range(4), range(4), range(3), range(3)], names=["Zone", "p", "q", "P", "Q"]
    ),
    columns=["AIC", "BIC"],
    dtype=np.float64(),
)


for (zone, p, q, P, Q), _ in selection.iterrows():

    print(zone, p, q, P, Q)

    model = sm.tsa.SARIMAX(
        pseudores.loc[zone, train_index].reset_index(drop=True),
        order=(p, 0, q),
        seasonal_order=(P, 0, Q, 24),
    )
    res = model.fit(disp=False)

    selection.loc[(zone, p, q, P, Q), "AIC"] = res.aic
    selection.loc[(zone, p, q, P, Q), "BIC"] = res.bic


# %% data up to time t-12 forecast t to t+24

n_sim = 1000
window = 24
offset = 12

forecasts = pd.DataFrame(
    columns=["Estimate", "Lower interval", "Upper interval"],
    index=pd.MultiIndex.from_product([zones, test_index]),
    dtype=np.float64,
)
# studentized residual
residuals = pd.Series(
    index=pd.MultiIndex.from_product([zones, train_index]),
    dtype=np.float64,
)
scores = pd.DataFrame(
    columns=["MAE", "RMSE", "CRPS", "VarS"],
    index=zones,
    dtype=np.float64,
)


params = []
variograms = {}

train_size = parameters["train_size"]
for zone in zones:

    print(zone)

    simulation = pd.DataFrame(
        columns=["Simulation" + str(x + 1) for x in range(n_sim)],
        index=test_index,
        dtype=np.float64,
    )

    forecast = pd.DataFrame(
        columns=["Estimate", "Lower interval", "Upper interval"],
        index=test_index,
        dtype=np.float64,
    )

    orders = selection["BIC"].groupby(level=0).idxmin()[zone]
    order = (orders[1], 0, orders[2])
    seasonal_order = (orders[3], 0, orders[4], 24)

    res = sm.tsa.SARIMAX(pseudores.loc[zone, train_index].values, order=order, seasonal_order=seasonal_order).fit()
    params.append(params_to_pd(res))
    residuals[zone] = res.standardized_forecasts_error

    res = res.apply(pseudores.loc[zone, index[: train_size - offset]].values, refit=False)

    for i in range(train_size, len(index), window):

        end = i + window
        if i + window >= len(index):
            end = len(index)

        save_idx = index[i:end]
        # n = len(save_idx)
        tmp = res.get_forecast(steps=len(save_idx) + offset)

        forecast.loc[save_idx, "Estimate"] = tmp.predicted_mean[offset:]
        forecast.loc[save_idx, ["Lower interval", "Upper interval"]] = tmp.conf_int(alpha=0.1)[offset:, :]

        simulation.loc[save_idx, :] = res.simulate(
            nsimulations=len(save_idx) + offset,
            repetitions=n_sim,
            anchor="end",
            random_state=rng,
        ).squeeze()[offset:, :]

        res = res.extend(pseudores.loc[zone, index[i - offset : end - offset]].values)

    model = nabqra.quantiles.spline_model(quantiles, *zone_lims[zone])
    quan = nabqra.misc.fix_quantiles(basis_quantiles.loc[zone, "Feature", test_index, :], *zone_lims[zone])

    forecast = model.back_transform(quan.values, forecast.values)[0]
    simulation = model.back_transform(quan.values, simulation)[0]

    scores.loc[zone, :] = calc_scores(obs.loc[zone, test_index].values, forecast[:, 0], simulation)
    forecasts.loc[zone, :, :] = forecast

    tmp = nabqra.scoring.variogram_distribution(simulation)
    variograms[f"{zone} expected variogram"] = tmp[0]
    variograms[f"{zone} std variogram"] = tmp[1]

params = pd.concat(params, axis=1, keys=zones)


# %% save data

selection.to_pickle(save_path / "model selection.pkl")
selection.to_csv(save_path / "model selection.csv")

forecasts.to_pickle(save_path / "Sarma forecasts.pkl")
forecasts.to_csv(save_path / "Sarma forecasts.csv")

scores.to_pickle(save_path / "Sarma scores.pkl")
scores.to_csv(save_path / "Sarma scores.csv")

residuals.to_pickle(save_path / "Sarma Residuals.pkl")
residuals.to_csv(save_path / "Sarma Residuals.csv")

params.to_pickle(save_path / "Params.pkl")
params.to_csv(save_path / "Params.csv")

np.savez(save_path / "Variograms", **variograms)


# %% with no extra data

n_sim = 1000
window = 24
offset = 12

forecasts = pd.DataFrame(
    columns=["Estimate", "Lower interval", "Upper interval"],
    index=pd.MultiIndex.from_product([zones, test_index]),
    dtype=np.float64,
)

scores = pd.DataFrame(
    columns=["MAE", "RMSE", "CRPS", "VarS"],
    index=zones,
    dtype=np.float64,
)


# params = []
variograms = {}

train_size = parameters["train_size"]
for zone in zones:

    print(zone)

    simulation = pd.DataFrame(
        columns=["Simulation" + str(x + 1) for x in range(n_sim)],
        index=test_index,
        dtype=np.float64,
    )

    forecast = pd.DataFrame(
        columns=["Estimate", "Lower interval", "Upper interval"],
        index=test_index,
        dtype=np.float64,
    )

    orders = selection["BIC"].groupby(level=0).idxmin()[zone]
    order = (orders[1], 0, orders[2])
    seasonal_order = (orders[3], 0, orders[4], 24)

    print("fit")
    res = sm.tsa.SARIMAX(pseudores.loc[zone, train_index].values, order=order, seasonal_order=seasonal_order).fit()
    # params.append(params_to_pd(res))
    # residuals[zone] = res.standardized_forecasts_error

    # res = res.apply(pseudores.loc[zone, index[: train_size - offset]].values, refit=False)

    print("sim")
    tmp = res.get_forecast(steps=window * 1000)
    sim = res.simulate(nsimulations=window * 1000, repetitions=n_sim, anchor="start", random_state=rng).squeeze()

    print("yes")
    for i in range(train_size, len(index), window):

        end = i + window
        if i + window >= len(index):
            end = len(index)

        save_idx = index[i:end]
        # n = len(save_idx)

        forecast.loc[save_idx, "Estimate"] = tmp.predicted_mean[-len(save_idx) :]
        forecast.loc[save_idx, ["Lower interval", "Upper interval"]] = tmp.conf_int(alpha=0.1)[-len(save_idx) :, :]

        simulation.loc[save_idx, :] = sim[-len(save_idx) :, :]

        # res = res.extend(pseudores.loc[zone, index[i - offset : end - offset]].values)
    print("transform")
    model = nabqra.quantiles.spline_model(quantiles, *zone_lims[zone])
    quan = nabqra.misc.fix_quantiles(basis_quantiles.loc[zone, "Feature", test_index, :], *zone_lims[zone])

    forecast = model.back_transform(quan.values, forecast.values)[0]
    simulation = model.back_transform(quan.values, simulation)[0]

    scores.loc[zone, :] = calc_scores(obs.loc[zone, test_index].values, forecast[:, 0], simulation)
    forecasts.loc[zone, :, :] = forecast

    tmp = nabqra.scoring.variogram_distribution(simulation)
    variograms[f"{zone} expected variogram"] = tmp[0]
    variograms[f"{zone} std variogram"] = tmp[1]

# params = pd.concat(params, axis=1, keys=zones)


# %%


forecasts.to_pickle(save_path / "Sarma nodata forecasts.pkl")
forecasts.to_csv(save_path / "Sarma nodata forecasts.csv")

scores.to_pickle(save_path / "Sarma nodata scores.pkl")
scores.to_csv(save_path / "Sarma nodata scores.csv")

np.savez(save_path / "Variograms nodata", **variograms)
