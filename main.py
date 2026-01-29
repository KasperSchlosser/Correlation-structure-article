# %%
import keras
import torch

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import scoringrules as sr

from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from keras import ops

# %% parameters
PATH = Path("C:/Users/kpfs/Projects/Correlation-structure-article")

#data_type = "Real"
data_type = "Simulated"

TRAINNN = True
#TRAINNN = False

# True to retrain models, false to use previous models
TRAINAR = True
#TRAINAR = False

quantiles = np.arange(0.05, 1.01, 0.05)
quantiles_str = [f"{x:.2f}" for x in quantiles]
order = (1, 0, 1)
seasonal_order = (1, 0, 0, 24)

timesteps = torch.flip(torch.arange(48, dtype=torch.int64), (-1,))
batch_size = 168
train_size = 1-0.46735  # 1 year of test data

epochs = 1000
nsim = 1000

zones = ["DK1-onshore", "DK2-onshore", "DK1-offshore", "DK2-offshore"]
zone_max = {
    "DK1-onshore": 3560,
    "DK2-onshore": 690,
    "DK1-offshore": 1370,
    "DK2-offshore": 990,
}

models_configs = {
    "Simple": keras.Sequential(
        [
            keras.layers.Rescaling(1 / 1500),
            keras.layers.GaussianDropout(1 / 100),
            keras.layers.Flatten(),
            keras.layers.Dense(1),
            keras.layers.Dense(len(quantiles), activation="softmax"),
        ]
    ).get_config(),
    "Latent": keras.Sequential(
        [
            keras.layers.Rescaling(1 / 1500),
            keras.layers.GaussianDropout(1 / 100),
            keras.layers.Dense(5, activation="selu"),
            keras.layers.Dense(5, activation="selu"),
            keras.layers.LSTM(5),
            keras.layers.Dense(5, activation="selu"),
            keras.layers.Dense(5, activation="selu"),
            keras.layers.Dense(len(quantiles), activation="softmax"),
        ]
    ).get_config(),
}


# %%
def load_observations(path, sep=";", decimal=","):

    data = pd.read_csv(path, sep=sep, decimal=decimal)
    data["Time"] = pd.to_datetime(data["HourUTC"], utc=True)

    data["offshore"] = (
        data["OffshoreWindLt100MW_MWh"] + data["OffshoreWindGe100MW_MWh"]
    )
    data["onshore"] = (
        data["OnshoreWindLt50kW_MWh"] + data["OnshoreWindGe50kW_MWh"]
    )

    # make columns
    data = data.pivot(
        columns="PriceArea", index="Time", values=["onshore", "offshore"]
    )
    data = data.swaplevel(0, 1, axis=1)
    data.columns = data.columns.map("-".join)

    # make into stacked
    data = data.stack(future_stack=True).swaplevel().sort_index()

    # remove zeros, gives problems later
    data.loc[data < 0.01] = 0.01

    return data


def load_ensembles(path):
    files = path.glob("*csv")

    dts = {}
    for file in files:
        zone = "-".join(file.stem.split("_")[0:3:2])
        dts[zone] = pd.read_csv(file, sep=";", decimal=",", index_col="date")
        dts[zone].index = pd.to_datetime(dts[zone].index, utc=True)

    dt = pd.concat(dts, keys=dts.keys(), names=["Zone", "Time"])

    return dt


class Quantileloss(keras.losses.Loss):
    def __init__(self, quantiles, range, **kwargs):
        super().__init__(**kwargs)
        self.range = range
        self.quantiles = torch.tensor(quantiles)
        self.neg_quantiles = self.quantiles - 1

    def call(self, y_true, y_pred):
        q_pred = self.range * ops.cumsum(y_pred, axis=-1)

        d = (y_true - q_pred.T).T
        x1 = self.quantiles * d
        x2 = self.neg_quantiles * d

        return ops.sum(ops.maximum(x1, x2), axis=-1)


class NABQRDataset(Dataset):

    def __init__(self, X, Y):

        self.Y = torch.tensor(Y)
        self.X = torch.tensor(X)

        self.timesteps = timesteps
        self.start = self.timesteps.max()

        if self.start > 0:
            pad = torch.zeros((self.start, X.shape[-1]))
            self.X = torch.cat((pad, self.X))

    def __len__(self):
        return self.Y.size(-1)

    def __getitem__(self, idx):

        return (
            torch.index_select(self.X, 0, idx - self.timesteps + self.start),
            self.Y[idx],
        )


def train_model(X, Y, config, loss):

    data = NABQRDataset(X, Y)
    train, val = [
        DataLoader(x, batch_size=batch_size)
        for x in random_split(data, [train_size, 1 - train_size])
    ]

    model = keras.Sequential.from_config(config)
    model.compile(loss=loss, optimizer=keras.optimizers.Adam())
    val_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.1,
        restore_best_weights=True,
    )

    hist = model.fit(
        train, validation_data=val, epochs=epochs, callbacks=[val_stop]
    )

    return model, hist


def apply_model(model, ensembles, observations, zone):

    data = DataLoader(
        NABQRDataset(ensembles.values, observations.values),
        batch_size=batch_size,
    )

    quants = (
        np.cumulative_sum(model.predict(data), axis=1, include_initial=True)
        * zone_max[zone]
    )

    return quants


def transform2normal(data, pred_quantiles):

    quants = np.pad(quantiles, (1, 0), "constant")

    xhat = np.zeros_like(data)
    for i in range(len(data)):
        xhat[i] = np.interp(data[i], pred_quantiles[i, :], quants)

    xhat = stats.norm.ppf(xhat)
    return xhat


def transform2orig(resid, pred_quantiles):

    quants = np.pad(quantiles, (1, 0), "constant")
    resid = stats.norm.cdf(resid)

    xhat = np.zeros_like(resid)
    for i in range(len(resid)):
        xhat[i, ...] = np.interp(resid[i, ...], quants, pred_quantiles[i, :])

    return xhat


def calc_scores(actuals, simulations, window_size=24, p=0.5):

    mu = np.median(simulations, axis=1)

    mae = np.mean(np.abs(actuals - mu))
    rmse = np.sqrt(np.mean((actuals - mu) ** 2))
    crps = sr.crps_ensemble(actuals, simulations).mean()

    # vars needs to be massaged

    n = len(actuals) // window_size
    k = simulations.shape[1]

    actuals = actuals[: n * window_size].reshape(-1, window_size)
    simulations = (
        simulations[: n * window_size, :]
        .reshape(-1, window_size, k)
        .transpose(0, -1, -2)
    )

    # mean score per "window" (i,j).
    # here ignoreing diagonal that is always 0
    vs = sr.vs_ensemble(actuals, simulations, p=p) / (
        window_size * (window_size - 1)
    )
    vs = vs.mean() ** (1 / (2 * p))

    return (mae, rmse, crps, vs)


def params_to_pd(res):
    data = np.array(res.summary().tables[1].data)
    return pd.DataFrame(
        data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:]
    )[["coef", "std err"]]

# %% load data

if data_type == "Real":
    observations = load_observations(
        PATH  / "Data" / data_type / "Raw Data" / "ProductionConsumptionSettlement.csv"
    )
    ensembles = load_ensembles(PATH  / "Data" / data_type / "Raw Data" / "Ensembles")
    observations = observations.loc[ensembles.index]

    pd_index = pd.MultiIndex.from_product(
        [zones, models_configs.keys(), observations.index.unique(1)]
    ).sortlevel()[0]

    train_idx = observations.index.unique(1)
    test_idx = train_idx[int(len(train_idx) * train_size) :]
    train_idx = train_idx[: int(len(train_idx) * train_size)]

    ensembles.to_pickle(PATH  / "Data" / data_type / "ensembles.pkl")
    observations.to_pickle(PATH  / "Data" / data_type / "observations.pkl")
else:
    ensembles = pd.read_pickle(PATH / "Data" / data_type / "ensembles.pkl")
    observations = pd.read_pickle(PATH / "Data" / data_type / "observations.pkl")

    pd_index = pd.MultiIndex.from_product(
        [zones, models_configs.keys(), observations.index.unique(1)]
    ).sortlevel()[0]

    train_idx = observations.index.unique(1)
    test_idx = train_idx[int(len(train_idx) * train_size) :]
    train_idx = train_idx[: int(len(train_idx) * train_size)]

# %%
models = {zone: {} for zone in zones}

if TRAINNN:

    history = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [zones, models_configs.keys(), np.arange(epochs + 1)]
        ).sortlevel()[0],
        columns=["Train loss", "Validation loss"],
    )
    for zone in zones:

        print(zone)
        obs_vals = observations.loc[zone].values
        ens_vals = ensembles.loc[zone].values

        loss = Quantileloss(quantiles, zone_max[zone])

        for name, config in models_configs.items():

            print(name)

            model, hist = train_model(ens_vals, obs_vals, config, loss)

            models[zone][name] = model
            model.save(PATH  / "Models" / data_type / "nn" / f"{zone}_{name}.keras")
            history.loc[zone, name, hist.epoch] = np.array(
                list(hist.history.values())
            ).T
    history.to_pickle(PATH  / "Data" / data_type / "modeltraining.pkl")
else:
    for file in (PATH  / "Models" / data_type / "nn").glob("*keras"):
        zone, name = file.stem.split("_")
        models[zone][name] = keras.saving.load_model(file, compile=False)

# %% marginal quantiles

marginal_quantiles = pd.DataFrame(
    index=pd_index, columns=["0.00"] + quantiles_str, dtype=np.float64
)

for zone in zones:
    for name, model in models[zone].items():
        marginal_quantiles.loc[zone, name, :] = apply_model(
            model, ensembles.loc[zone], observations.loc[zone], zone
        )

marginal_quantiles.to_pickle(PATH  / "Data" / data_type / "marginal_quantiles.pkl")

# %% pesudo residuals
pseudoresid = pd.Series(index=pd_index, dtype=np.float64)
for zone in zones:
    for name in models_configs.keys():
        tmp = transform2normal(
            observations[zone].values,
            marginal_quantiles.loc[zone, name].values,
        )
        pseudoresid.loc[zone, name] = tmp

pseudoresid.to_pickle(PATH  / "Data" / data_type / "pseudoresidual.pkl")

# %% arima estimation and simulation

simulations_normal = pd.DataFrame(
    index=pd_index,
    columns=[f"Simulation {x}" for x in range(1, nsim + 1)],
    dtype=np.float64,
)
residuals_normal = pd.Series(index=pd_index, dtype=np.float64)

parameters = {zone: {} for zone in zones}

for zone in zones:
    for name in models_configs.keys():
        print(f"{zone}-{name}")

        p = PATH  / "Models" / data_type / "sarima" / f"{zone}_{name}.npy"
        if TRAINAR:
            model = sm.tsa.SARIMAX(
                pseudoresid.loc[zone, name, train_idx].values,
                order=order,
                seasonal_order=seasonal_order,
            ).fit()

            parameters[zone][name] = params_to_pd(model)
            params = model.params

            np.save(p, params)

            model = model.apply(pseudoresid.loc[zone, name].values, refit = False)
            residuals_normal.loc[zone,name] = model.resid

        else:
            params = np.load(p)
        model = sm.tsa.SARIMAX(
            np.zeros(30), order=order, seasonal_order=seasonal_order
        )

        simulations_normal.loc[zone, name] = model.simulate(
            params,
            nsimulations=len(observations.index.unique(1)),
            repetitions=nsim,
            anchor="start",
        ).squeeze()

if TRAINAR:
    parameters = pd.concat(
        [
            pd.concat(parameters[zone].values(), keys=parameters[zone].keys())
            for zone in zones
        ],
        keys=zones,
    )
    parameters.to_pickle(PATH  / "Data" / data_type / "corrolation_parameters.pkl")
    residuals_normal.to_pickle(PATH  / "Data" / data_type / "residuals_normal.pkl")

simulations_normal.to_pickle(PATH  / "Data" / data_type / "simulation_normal.pkl")


# %% original space

simulations = pd.DataFrame(
    index=pd_index,
    columns=[f"Simulation {x}" for x in range(1, nsim + 1)],
    dtype=np.float64,
)

#for plotting take quantiles over last 2400 observations,
simulations_plot = pd.DataFrame(
    index=pd_index,
    columns=quantiles_str,
    dtype=np.float64,
)

for zone in zones:
    for name in models_configs.keys():
        print(f"{zone}-{name}")
        simulations.loc[zone, name] = transform2orig(
            simulations_normal.loc[zone, name],
            marginal_quantiles.loc[zone, name].values,
        )

        tmp = np.quantile(simulations_normal.loc[zone,name].iloc[-2400:], quantiles)
        tmp = np.broadcast_to(tmp, (simulations_normal.loc[zone,name].shape[0], len(tmp)))
        simulations_plot.loc[zone, name] = transform2orig(
            tmp,
            marginal_quantiles.loc[zone, name].values,
        )
simulations.to_pickle(PATH  / "Data" / data_type / "simulation.pkl")
simulations_plot.iloc[:,:-1].to_pickle(PATH  / "Data" / data_type / "simulations_plot.pkl")

# %% scores

model_names = (
    ["Ensembles"]
    + [k + " - Marginal" for k in models_configs.keys()]
    + list(models_configs.keys())
)

scores = pd.DataFrame(
    index=pd.MultiIndex.from_product([zones, model_names]),
    columns=["MAE", "RMSE", "CRPS", "VarS"],
    dtype=np.float64,
)

for zone in zones:
    print(zone)
    for name in models_configs.keys():
        tmp = calc_scores(
            observations.loc[zone, test_idx].values,
            simulations.loc[zone, name, test_idx].values,
        )
        scores.loc[zone, name, :] = tmp

# marginal simulations - no correlation structure
sim = stats.norm().rvs((len(test_idx), nsim))
for zone in zones:
    print(zone)
    for name in models_configs.keys():
        vals = transform2orig(
            sim, marginal_quantiles.loc[zone, name, test_idx].values
        )
        tmp = calc_scores(observations.loc[zone, test_idx].values, vals)
        scores.loc[zone, name + " - Marginal", :] = tmp

# original ensembles
for zone in zones:
    tmp = calc_scores(
        observations.loc[zone, test_idx].values,
        ensembles.loc[zone, test_idx, :].values,
    )
    scores.loc[zone, "Ensembles"] = tmp

scores_percent = scores.divide(scores.xs("Ensembles", level=1), level=0)

scores.to_pickle(PATH  / "Data" / data_type / "scores.pkl")
scores_percent.to_pickle(PATH  / "Data" / data_type / "scores_percent.pkl")

