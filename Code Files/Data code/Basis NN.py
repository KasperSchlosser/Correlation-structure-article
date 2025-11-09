import pathlib
import tomllib

import pandas as pd
import numpy as np

import keras

from itertools import product
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from nabqra.scoring import Quantileloss
from nabqra.misc import NABQRDataset

# from pandas import IndexSlice as idx

keras.utils.set_random_seed(666)

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Data"
save_path = PATH / "Data" / "Basis"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]
    train_size = parameters["train_size"]

# load input
ensembles = pd.read_pickle(load_path / "cleaned_ensembles.pkl")

observations = pd.read_pickle(load_path / "cleaned_observations.pkl")
obs_max = observations.groupby("Zone").max()
obs_scaled = observations / obs_max  # scale to help optimizer

date_index = ensembles.index.unique(level=1)

quantiles_str = [f"{x:.2f}" for x in parameters["Quantiles"]]
Loss = Quantileloss(parameters["Quantiles"])


# %% define models
models = [
    {
        "name": "NABQR",
        "model_config": keras.Sequential(
            [
                keras.Input(shape=(7, ensembles.shape[-1])),
                keras.layers.LSTM(256, return_sequences=False),
                keras.layers.Dense(len(parameters["Quantiles"]), activation="sigmoid"),
                keras.layers.Dense(len(parameters["Quantiles"]), activation="leaky_relu"),
            ]
        ).get_config(),
        "opt_args": {"learning_rate": 1e-3},
        "data_args": {"reverse": False, "timesteps": (0, 1, 2, 6, 12, 24, 48), "preprocessing": MinMaxScaler()},
        "loader_args": {
            "batch_size": 24 * 7,
        },
    },
    {
        "name": "Simple",
        "model_config": keras.Sequential(
            [
                keras.Input(shape=(1, ensembles.shape[-1])),
                keras.layers.Dense(1),
                keras.layers.Dense(len(parameters["Quantiles"])),
            ]
        ).get_config(),
        "opt_args": {"learning_rate": 1e-3},
        "data_args": {"reverse": True, "timesteps": (0,), "preprocessing": MinMaxScaler(feature_range=(-1, 1))},
        "loader_args": {"batch_size": 24 * 7, "shuffle": False},
    },
    {
        "name": "Feature",
        "model_config": keras.Sequential(
            [
                keras.Input(shape=(49, ensembles.shape[-1])),
                keras.layers.GaussianDropout(1 / 100),
                keras.layers.Dense(5, activation="selu"),
                keras.layers.Dense(5, activation="selu"),
                keras.layers.LSTM(5),
                keras.layers.Dense(5, activation="selu"),
                keras.layers.Dense(5, activation="selu"),
                keras.layers.Dense(len(parameters["Quantiles"])),
            ]
        ).get_config(),
        "opt_args": {"learning_rate": 1e-3},
        "data_args": {
            "reverse": True,
            "timesteps": np.arange(49),
            "preprocessing": MinMaxScaler(feature_range=(-1, 1)),
        },
        "loader_args": {"batch_size": 24 * 7, "shuffle": False},
    },
]

# %% train and predict

history = {z: {} for z in zones}
predictions = {z: {} for z in zones}
features = pd.DataFrame(index=ensembles.index, columns=[f"Feature {x+1}" for x in range(5)], dtype=np.float64)

for zone, vals in product(zones, models):

    name = vals["name"]
    print(zone, name)

    # for the test data  i need to pass in slightly more data
    # this is because the first index is where it is possible to get the  full data
    # ie if the models needs data from lag 48 the first data point is at 48
    # this just ensures all models test on the same data

    # inputs need to be preprocceseds
    if "preprocessing" in vals["data_args"]:
        vals["data_args"]["preprocessing"].fit(ensembles.loc[zone][train_size:].values)

    full_data = NABQRDataset(ensembles.loc[zone].values, obs_scaled.loc[zone].values, **vals["data_args"])
    full_data = DataLoader(full_data, **vals["loader_args"])

    train_data = NABQRDataset(
        ensembles.loc[zone].values[:train_size], obs_scaled.loc[zone].values[:train_size], **vals["data_args"]
    )
    train_data = DataLoader(train_data, **vals["loader_args"])

    test_data = NABQRDataset(
        ensembles.loc[zone].values[train_size:], obs_scaled.loc[zone].values[train_size:], **vals["data_args"]
    )
    test_data = DataLoader(test_data, **vals["loader_args"])

    model = keras.Sequential.from_config(vals["model_config"])
    model.compile(loss=Quantileloss(parameters["Quantiles"]), optimizer=keras.optimizers.Adam(**vals["opt_args"]))
    val_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    hist = model.fit(train_data, validation_data=test_data, epochs=1000, callbacks=[val_stop])
    preds = model.predict(full_data)

    history[zone][name] = pd.DataFrame(hist.history)
    history[zone][name].index.name = "Epoch"

    predictions[zone][name] = pd.DataFrame(preds.squeeze(), index=date_index, columns=quantiles_str)

    if name == "Feature":
        extractor = keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
        features.loc(axis=0)[zone, :] = extractor.predict(full_data).squeeze()

    # save the models

    model.save(save_path / "Models" / f"{name} - {zone}.keras")


# %% save results
# maybe i should save the models?
history = pd.concat(
    {
        k: pd.concat(
            v,
            names=[
                "Model",
            ],
        )
        for k, v in history.items()
    },
    names=["Zone"],
)
predictions = pd.concat(
    {
        k: pd.concat(
            v,
            names=[
                "Model",
            ],
        )
        for k, v in predictions.items()
    },
    names=["Zone"],
)
predictions_original = predictions.mul(obs_max, axis=0)


# %%
history.to_csv(save_path / "History.csv")
history.to_pickle(save_path / "History.pkl")

predictions.to_csv(save_path / "Basis normalised.csv")
predictions.to_pickle(save_path / "Basis normalised.pkl")
predictions_original.to_csv(save_path / "NN quantiles.csv")
predictions_original.to_pickle(save_path / "NN quantiles.pkl")

features.to_csv(save_path / "Features.csv")
features.to_pickle(save_path / "Features.pkl")
