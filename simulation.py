# code taken from NABQR https://github.com/bast0320/nabqr
#
import numpy as np
import pandas as pd

from pathlib import Path

PATH = Path("C:/Users/kpfs/Projects/Correlation-structure-article")

def simulate_ou(mu, kappa, sigma, n_ensembles):

    X = np.zeros((len(mu), n_ensembles))
    X[0,:] = mu[0]  # initial condition

    for t in range(1, len(mu)):
        dW = np.random.normal(0,1, n_ensembles)
        X[t] = X[t - 1] + kappa * (mu[t] - X[t - 1])  + sigma * dW

    return X

#%%
if __name__ == "__main__":

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
        ens = simulate_ou(z.values, kappa = 0.33, sigma = 0.033, n_ensembles = 52)

        ensembles.loc[zone] = ens * zone_max[zone]

    ensembles.to_pickle("Data/Simulated/ensembles.pkl")
    observations.to_pickle("Data/Simulated/observations.pkl")



