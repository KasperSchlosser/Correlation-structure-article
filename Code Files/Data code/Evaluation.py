import pathlib
import tomllib


import pandas as pd
import numpy as np
import statsmodels.api as sm
import numba as nb

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data"
save_path = PATH / "Data" / "Evaluation"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]

rng = np.random.default_rng(42)


@nb.jit(nb.float64[:, :](nb.float64[:, :], nb.float64), nopython=True, nogil=True, parallel=True)
def vars_e(arr, p=0.5):

    n, k = arr.shape
    res = np.zeros((n, n))

    for col in range(k):
        for i in range(n):
            for j in range(n):
                res[i, j] += abs(arr[i, col] - arr[j, col]) ** p

    for i in range(n):
        for j in range(i + 1, n):
            res[j, i] = res[i, j]
    return res / k


@nb.jit(nb.float64[:, :, :](nb.float64[:, :], nb.float64[:, :], nb.float64), nopython=True, nogil=True, parallel=True)
def VarS(arr, mu, p):

    n, k = arr.shape

    res = np.zeros((k, n, n))

    for col in range(k):
        for i in range(n):
            for j in range(i + 1, n):
                res[col, i, j] += (abs(arr[i, col] - arr[j, col]) ** p - mu[i, j]) ** 2

    for i in range(n):
        for j in range(i + 1, n):
            res[:, j, i] = res[:, i, j]
    # equal waight but no weight for diagonal
    return res / (n * (n - 1))


# %%

mod = sm.tsa.SARIMAX([0] * 24, order=(2, 0, 0))
params = np.array([0.8, -0.64, 1])
params_wrong = np.array([0.8, 0, 1])
offset = 2
ps = [0.5, 1, 2]
quantiles = [0.5, 0.05, 0.95]


# %%
sims_correct = mod.simulate(params, 24, initial_state=[offset] + [0] * 1, repetitions=10000).squeeze()
sims_wrong = mod.simulate(params_wrong, 24, initial_state=[offset] + [0] * 1, repetitions=10000).squeeze()

process = pd.concat(
    [
        pd.DataFrame(np.quantile(sims_correct, quantiles, axis=1).T, columns=[f"{x:.2f}" for x in quantiles]),
        pd.DataFrame(np.quantile(sims_wrong, quantiles, axis=1).T, columns=[f"{x:.2f}" for x in quantiles]),
    ],
    keys=["AR(2) model", "AR(1) model"],
)


# %%
process.to_pickle(save_path / "process.pkl")
process.to_csv(save_path / "process.csv")


# %% weight vs. no weight

n = 24
k = 100000
sims_correct = mod.simulate(params, 10 * n, initial_state=[0] + [0] * 1, repetitions=k).squeeze()[9 * n :, :]
sims_wrong = mod.simulate(params_wrong, 10 * n, initial_state=[0] + [0] * 1, repetitions=k).squeeze()[9 * n :, :]

weight = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        weight[i, j] = 1 / np.abs(i - j)
        weight[j, i] = 1 / np.abs(i - j)

weight = weight / weight.sum()

# %% order and score

scores_df = pd.DataFrame(columns=["0.5", "1", "2"], index=range(sims_correct.shape[1]), dtype=np.float64)


for p in [0.5, 1, 2]:
    evars = vars_e(sims_correct, p)
    scores_df.loc[:, str(p)] = VarS(sims_correct, evars, p).sum(axis=(1, 2)) ** (1 / (2 * p))

scores_df.to_pickle(save_path / "scores.pkl")
scores_df.to_csv(save_path / "scores.csv")

# %% weight and scores
p = 0.5
e_correct = vars_e(sims_correct, p)
e_wrong = vars_e(sims_wrong, p)

scores_correct = VarS(sims_correct, e_correct, p)
scores_wrong = VarS(sims_correct, e_wrong, p)

scores_correct_weight = VarS(sims_correct, e_correct, p) * (n * (n - 1)) * weight
scores_wrong_weight = VarS(sims_correct, e_wrong, p) * (n * (n - 1)) * weight

diffs = scores_wrong - scores_correct
diffs_weight = scores_wrong_weight - scores_correct_weight

scores = diffs.sum(axis=(1, 2))
scores_weight = diffs_weight.sum(axis=(1, 2))

print((scores < 0).sum(), (scores_weight < 0).sum())

np.savez(
    save_path / "weight comp.npz",
    weight=weight,
    scores_correct=scores_correct,
    scores_wrong=scores_wrong,
    scores_correct_weight=scores_correct_weight,
    scores_wrong_weight=scores_wrong_weight,
    diffs=diffs,
    diffs_weight=diffs_weight,
    scores=scores,
    scores_weight=scores_weight,
    e_correct=e_correct,
    e_wrong=e_wrong,
)
