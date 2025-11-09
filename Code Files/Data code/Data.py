# loads raw data, cleans and normalize it

import pandas as pd
from pathlib import Path
import tomllib


def cleantz(ensemble):
    # drops timezones from the ensemble data
    # then drops the duplicated timeslot
    # might not be the best solution

    ensemble.index = ensemble.index.to_series().apply(lambda x: x.tz_localize(None)).values
    ensemble = ensemble[~ensemble.index.duplicated()]
    ensemble.dropna()

    return ensemble


PATH = Path.cwd().parents[1]
load_path = PATH / "Data" / "Raw Data"
save_path = PATH / "Data" / "Data"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)


# %% Load Data
# Finds ensembles and observations

ensembles = list((load_path / "Ensembles").glob("*pkl"))
observations = list((load_path / "Observations").glob("*pkl"))

raw_ensembles = {}
raw_observations = {}
cleaned_ensembles = {}
cleaned_observations = {}

for ens, obs in zip(ensembles, observations):

    zone = "-".join(ens.stem.split("_")[:2])
    print(zone)

    ens = pd.read_pickle(ens)

    obs = pd.read_pickle(obs)
    obs.name = "Observed"

    # need original data for plots
    raw_ensembles[zone] = ens.copy(deep=True)
    raw_observations[zone] = obs.copy(deep=True)

    # remove tz from ensemble and drop the missing hours from observation
    ens = cleantz(ens)

    # change problematic observations
    obs[obs < parameters["low_value"]] = parameters["low_value"]

    # removes rows at times skipped by daylight-savings
    obs = obs[ens.index]

    cleaned_ensembles[zone] = ens
    cleaned_observations[zone] = obs


# %%
cleaned_ensembles = pd.concat(cleaned_ensembles, keys=cleaned_ensembles.keys(), names=("Zone", "Time"))
cleaned_observations = pd.concat(cleaned_observations, keys=cleaned_observations.keys(), names=("Zone", "Time"))

raw_ensembles = pd.concat(raw_ensembles, keys=raw_ensembles.keys(), names=("Zone", "Time"))
raw_observations = pd.concat(raw_observations, keys=raw_observations.keys(), names=("Zone", "Time"))


cleaned_ensembles.to_pickle(save_path / "cleaned_ensembles.pkl")
cleaned_ensembles.to_csv(save_path / "cleaned_ensembles.csv")

cleaned_observations.to_pickle(save_path / "cleaned_observations.pkl")
cleaned_observations.to_csv(save_path / "cleaned_observations.csv")

raw_ensembles.to_pickle(save_path / "raw_ensembles.pkl")
raw_ensembles.to_csv(save_path / "raw_ensembles.csv")

raw_observations.to_pickle(save_path / "raw_observations.pkl")
raw_observations.to_csv(save_path / "raw_observations.csv")
