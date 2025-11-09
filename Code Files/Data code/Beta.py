import pathlib
import tomllib

import pandas as pd
import numpy as np
import statsmodels.othermod.betareg as betareg
import scipy.stats as stats

from pandas import IndexSlice as idx

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "NABQR"
save_path = PATH / "Data"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]


def trans(x, zone):
    f = zone_limits[zone]
    return x / f[1]

def detrans(x):
    f = zone_limits[zone]
    return f[1]*x

quantiles = np.concat([[0.01], np.arange(0.05, 1, 0.05), [0.99]])


actuals = pd.read_pickle(load_path / "actuals.pkl")
actuals[actuals < 1] = 1

for zone in zones:

    actuals[zone] = trans(actuals[zone].values, zone)

basis = pd.read_pickle(load_path / "corrected_ensembles.pkl")
basis.loc[:, "const"] = 1

# train_size = int(len(actuals)*0.64)
# X_train = basis.iloc[:train_size]
# y_train = actuals.iloc[:train_size]
# X_test = basis.iloc[train_size:]
# y_test = actuals.iloc[train_size:]


beta_observations = pd.DataFrame(index=actuals.index,
                                 columns=pd.MultiIndex.from_product((
                                     zones,
                                     ["CDF", "Normal"]
                                 )),
                                 dtype=np.float64
                                 )
beta_quantiles = pd.DataFrame(index=actuals.index,
                              columns=pd.MultiIndex.from_product((
                                  zones,
                                  [f'{q:.2f}' for q in quantiles]
                              )),
                              dtype=np.float64
                              )
# %%
for zone in zones:
    print(zone)

    # qs = [f'{q:.2f}' for q in quantiles[::2]]
    qs = [f'{q:.2f}' for q in quantiles]
    reduced_X = basis.loc[:, idx[zone, qs]].values
    reduced_basis = basis.loc[:, idx[zone, qs]].values
    # model = betareg.BetaModel(y_train[zone], X_train[zone], exog_precision=X_train[zone]).fit()
    # model = betareg.BetaModel(actuals[zone], reduced_X, exog_precision=reduced_X).fit()
    model = betareg.BetaModel(actuals[zone], reduced_X).fit()

    # dist = model.get_distribution(basis[zone], exog_precision=basis[zone])
    dist = model.get_distribution(reduced_basis)
    dist = model.get_distribution(reduced_basis)

    print("done training")
    beta_quantiles[zone] = detrans(dist.ppf(quantiles[:, np.newaxis]).T)
    beta_observations[zone, "CDF"] = dist.cdf(actuals[zone].values.T).T
    beta_observations[zone, "Normal"] = stats.norm().ppf(beta_observations[zone, "CDF"])

beta_quantiles.to_csv(save_path / "NABQR" / "beta_quantiles.csv")
beta_quantiles.to_pickle(save_path / "NABQR" / "beta_quantiles.pkl")

beta_observations.to_csv(save_path / "Distribution" / "beta_observations.csv")
beta_observations.to_pickle(save_path / "Distribution" / "beta_observations.pkl")
