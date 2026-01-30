# code taken from NABQR https://github.com/bast0320/nabqr
#
import numpy as np
import pandas as pd

from pathlib import Path

PATH = Path("C:/Users/kpfs/Projects/Correlation-structure-article")

def simulate_ou(mu, n_ensembles, kappa=0.25, lamb = 0.95):

    X = np.zeros((len(mu), n_ensembles))
    sigmas = np.zeros((len(mu)))
    X[0,:] = mu[0]  # initial condition
    sigmas[0] = 0.1**2

    for t in range(1, len(mu)):
        dW = np.random.normal(0,1, n_ensembles)
        sigmas[t] = lamb * sigmas[t - 1] + (1-lamb)*(mu[t] - mu[t - 1])**2
        X[t] = np.clip(X[t - 1] + kappa * (mu[t] - X[t - 1])  + np.sqrt(sigmas[t]) * dW, 0,1)

    return X, sigmas

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    zones = ["DK1-onshore", "DK2-onshore", "DK1-offshore", "DK2-offshore"]
    zone_max = {
        "DK1-onshore": 3560,
        "DK2-onshore": 690,
        "DK1-offshore": 1370,
        "DK2-offshore": 990,
    }

    # Your existing code
    observations = pd.read_pickle(PATH / "Data" / "Real" / "observations.pkl")
    ensembles = pd.DataFrame(index = observations.index, columns = range(52), dtype=float)


    for zone in observations.index.unique(0):

        z = observations.loc[zone] / zone_max[zone]
        ens, sigmas = simulate_ou(z.values, kappa = 0.1, n_ensembles = 52)

        ensembles.loc[zone] = ens * zone_max[zone]

        plt.figure()
        plt.plot(sigmas)
        plt.savefig(f"tmp_sigma_{zone}.png")


    ensembles.to_pickle("Data/Simulated/ensembles.pkl")
    observations.to_pickle("Data/Simulated/observations.pkl")

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), squeeze=True)
    example_date = pd.to_datetime(["2023-07-01", "2023-07-14"], utc=True)
    axes = axes.ravel()
    for i, zone in enumerate(zones):
        tmp_ens = ensembles.loc[zone].loc[example_date[0]: example_date[1]]
        tmp_obs = observations.loc[zone].loc[example_date[0]: example_date[1]]
        axes[i].plot(tmp_ens.index, tmp_ens, color="black", alpha=0.05, label="Ensemble")
        axes[i].scatter(tmp_obs.index, tmp_obs, s=1, label="Actual Production")
        axes[i].set_title(zone)
        axes[i].tick_params(axis="x", labelrotation=20)

    axes[2].set_xlim(*example_date)
    fig.supxlabel("Date")
    fig.supylabel("MWh")
    fig.set_layout_engine("constrained")

    fig.savefig("tmp.png")

    plt.figure()
    plt.plot()


