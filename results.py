#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

PATH = Path("C:/Users/kpfs/Projects/Correlation-structure-article")

zones = ["DK1-onshore", "DK2-onshore", "DK1-offshore", "DK2-offshore"]
example_date = pd.to_datetime(["2023-07-01", "2023-07-14"])

# raw data
ens = pd.read_pickle(PATH / "Data" / "ensembles.pkl")
obs = pd.read_pickle(PATH / "Data" / "observations.pkl")


fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), squeeze=True)
axes = axes.ravel()
for i, zone in enumerate(zones):
    axes[i].plot(ens.loc[zone].index, ens.loc[zone], color="black", alpha=0.05)
    axes[i].scatter(obs.loc[zone].index, obs.loc[zone], s=1)

    axes[i].set_title(zone)
    axes[i].tick_params(axis="x", labelrotation=30)

axes[2].set_xlim(*example_date)
fig.supxlabel("Date")
fig.supylabel("MWh")
fig.set_layout_engine("constrained")

# %%

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

fig.supxlabel("Epoch")
fig.supylabel("Quantile Loss")
fig.set_layout_engine("constrained")
