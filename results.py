# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

from pathlib import Path

idx = pd.IndexSlice
PATH = Path("C:/Users/kpfs/Projects/Correlation-structure-article")

zones = ["DK1-onshore", "DK2-onshore", "DK1-offshore", "DK2-offshore"]
example_date = pd.to_datetime(["2023-07-01", "2023-07-14"], utc=True)

# %% raw data
ens = pd.read_pickle(PATH / "Data" / "Real" / "ensembles.pkl")
obs = pd.read_pickle(PATH / "Data" / "Real" / "observations.pkl")


fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), squeeze=True)
axes = axes.ravel()
for i, zone in enumerate(zones):

    tmp_ens = ens.loc[zone].loc[example_date[0] : example_date[1]]
    tmp_obs = obs.loc[zone].loc[example_date[0] : example_date[1]]
    axes[i].plot(
        tmp_ens.index,
        tmp_ens,
        color="black",
        alpha=0.05,
        label="Ensemble",
    )
    axes[i].scatter(tmp_obs.index, tmp_obs, s=1, label="Actual Production")

    axes[i].set_title(zone)
    axes[i].tick_params(axis="x", labelrotation=20)

axes[2].set_xlim(*example_date)
fig.supxlabel("Date")
fig.supylabel("MWh")
fig.set_layout_engine("constrained")

fig.savefig(PATH / "Results" / "Graphs" / "data_example.pdf")

train_size = 1-0.46735
timepoints = obs.index.get_level_values(1).unique()
print(f"Period:{timepoints.min()} {timepoints.max()} Total data: {len(timepoints)} hours")
print(f"split at {train_size:.2%} -> {int(len(timepoints)*train_size)} hours train, {int(len(timepoints)*(1-train_size))} hours test")
print(f"Train period: {timepoints.min()} to {timepoints[int(len(timepoints)*train_size)-1]}")

# %% data summary

tmp = obs.unstack(level=0).describe().drop("count")
tmp.style.format(precision=2).format_index(escape="latex").to_latex(
    PATH / "Results" / "Tables" / "observations_summary.tex",
    position="htb",
    hrules=True,
    caption=(
        "Summary Statistics for the observed power productions",
        "Observation summary",
    ),
    label="tab:observation",
    position_float="centering",
)

tmp = (
    ens.unstack(level=0)
    .stack(level=0, future_stack=True)
    .describe()
    .drop("count")
)
tmp.style.format(precision=2).format_index(escape="latex").to_latex(
    PATH / "Results" / "Tables" / "ensembles_summary.tex",
    position="htb",
    hrules=True,
    caption=(
        "Summary Statistics for the Ensembles",
        "Ensemble summary",
    ),
    label="tab:ensemble",
    position_float="centering",
)

# %% training loss

train = pd.read_pickle(PATH / "Data" / "Real" / "modeltraining.pkl")

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


axes[1].plot(0, 0, color="black", label="Train Loss")
axes[1].plot(0, 0, color="black", ls="--", label="Validation Loss")
axes[1].legend()

fig.supxlabel("Epoch")
fig.supylabel("Quantile Loss")
fig.set_layout_engine("constrained")

fig.savefig(PATH / "Results" / "Graphs" / "loss_curves.pdf")

# %% Marginal models

marginal_quantiles = pd.read_pickle(PATH / "Data" / "Real" / "marginal_quantiles.pkl")

fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), squeeze=True)
axes = axes.ravel()
for i, zone in enumerate(zones):
    q = marginal_quantiles.loc[zone, "Latent"]
    x = q.index

    l = axes[i].plot(x, q["0.50"])

    for j in range(1, len(marginal_quantiles.columns) // 2):
        axes[i].fill_between(
            x,
            q.iloc[:, j],
            q.iloc[:, -j - 1],
            color=l[0].get_color(),
            alpha=0.2,
        )

    # Observations
    axes[i].scatter(obs.loc[zone].index, obs.loc[zone], s=3, color="black")

    axes[i].set_title(zone)
    axes[i].tick_params(axis="x", labelrotation=20)

axes[2].set_xlim(*example_date)
fig.supxlabel("Date")
fig.supylabel("MWh")
fig.set_layout_engine("constrained")

fig.savefig(PATH / "Results" / "Graphs" / "marginal_example.pdf")


# %% pseudo residual

marginal_quantiles = pd.read_pickle(PATH / "Data" / "Real" / "marginal_quantiles.pkl")
pseudo = pd.read_pickle(PATH / "Data" / "Real" / "pseudoresidual.pkl")

fig, ax = plt.subplots(figsize=(8, 6))

pr = pseudo.loc["DK1-onshore", "Latent"]
x = pr.index

l = ax.axhline(0, label="Median")

for i in range(1, len(marginal_quantiles.columns) // 2):
    q1 = stats.norm.ppf(float(marginal_quantiles.columns[i]))
    q2 = stats.norm.ppf(float(marginal_quantiles.columns[-i - 1]))
    ax.axhspan(q1, q2, color=l.get_color(), alpha=0.2)

# Observations
ax.scatter(pr.index, pr, s=3, color="black", label="Actual production")

ax.set_title("DK1-onshore")
ax.set_xlabel("Date")
ax.set_ylabel("MWh")
ax.tick_params(axis="x", labelrotation=20)
ax.set_xlim(*example_date)

fig.set_layout_engine("constrained")
fig.savefig(PATH / "Results" / "Graphs" / "pseudoresidual.pdf")


# %% arima params

params = pd.DataFrame(
    columns=pd.MultiIndex.from_product([zones, ["Latent", "Simple"]]),
    index=[r"$\theta_1$", r"$\phi_1$", r"$\Theta_1$", r"$\sigma^2$"],
    dtype=np.float64,
)
for f in (PATH / "Models" / "Real" / "sarima").glob("*"):

    zone, model = f.stem.split("_")
    data = np.load(f)

    params[zone, model] = data

params.style.format(precision=2).to_latex(
    PATH / "Results" / "Tables" / "parameters.tex",
    hrules=True,
    clines="all;data",
)

# %% acf and pacf plots
pseudo = pd.read_pickle(PATH / "Data" / "Real" / "pseudoresidual.pkl")
residuals = pd.read_pickle(PATH / "Data" / "Real" / "residuals_normal.pkl")

width = 0.1
nlags=48
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6), squeeze=True)
for i, zone in enumerate(zones):
    X = np.arange(49) + 0.25*(i - 2)
    acf_before = sm.tsa.acf(pseudo.loc[zone, "Latent"], nlags=nlags)
    pacf_before = sm.tsa.pacf(pseudo.loc[zone, "Latent"], nlags=nlags)

    acf_after = sm.tsa.acf(residuals.loc[zone, "Latent"], nlags=nlags)
    pacf_after = sm.tsa.pacf(residuals.loc[zone, "Latent"], nlags=nlags)

    axes[0, 0].bar(X, acf_before, width=width, label=zone)
    axes[0, 1].bar(X, pacf_before, width=width, label=zone)
    axes[1, 0].bar(X, acf_after, width=width, label=zone)
    axes[1, 1].bar(X, pacf_after, width=width, label=zone)

fig.supxlabel("Lag")
axes[0,0].set_title("ACF")
axes[0,1].set_title("PACF")

axes[0,0].set_ylabel("Before")
axes[1,0].set_ylabel("after")

axes[0,1].legend()

fig.set_layout_engine("constrained")

fig.savefig(PATH / "Results" / "Graphs" / "acf_pacf.pdf")


# %% Autocorrelation models

sims = pd.read_pickle(PATH / "Data" / "Real" / "simulations_plot.pkl")

fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), squeeze=True)
axes = axes.ravel()
for i, zone in enumerate(zones):
    q = sims.loc[zone, "Latent"]
    x = q.index

    l = axes[i].plot(x, q["0.50"])

    for j in range(1, len(marginal_quantiles.columns) // 2):
        axes[i].fill_between(
            x,
            q.iloc[:, j],
            q.iloc[:, -j - 1],
            color=l[0].get_color(),
            alpha=0.2,
        )

    # Observations
    axes[i].scatter(obs.loc[zone].index, obs.loc[zone], s=3, color="black")

    axes[i].set_title(zone)
    axes[i].tick_params(axis="x", labelrotation=20)

axes[2].set_xlim(*example_date)
fig.supxlabel("Date")
fig.supylabel("MWh")
fig.set_layout_engine("constrained")

fig.savefig(PATH / "Results" / "Graphs" / "corrolation_example.pdf")


# %% Scores

scores = pd.read_pickle(PATH / "Data" / "Real" / "scores.pkl")
scores_percent = pd.read_pickle(PATH / "Data" / "Real" / "scores_percent.pkl")

is_ensemble = scores.index.get_level_values(1) == "Ensembles"
combined = scores_percent.copy()
combined.loc[is_ensemble] = scores.loc[is_ensemble]

# --- Styling / LaTeX ---

def pct_fmt(x):
    # Handle NaNs gracefully for LaTeX
    return f"{x * 100:.1f}\\%" if pd.notna(x) else ""

def num_fmt(x):
    return f"{x:.2f}" if pd.notna(x) else ""

styler = (
    combined.style
    .format(num_fmt, subset=idx[is_ensemble, :], escape="latex")
    .format(pct_fmt, subset=idx[~is_ensemble, :], escape="latex")
    .highlight_between(
        left=0, right=1, inclusive="neither",
        subset=idx[~is_ensemble, :],
        props="font-weight:bold;"
    )
)
styler.to_latex(
    PATH / "Results" / "Tables" / "scores.tex",
    hrules=True,
    clines="all;data",
    position="htb",
    caption=(
        "Model Scores (Ensemble in Original Units; Other Models as Percent of Ensemble)",
        "Ensemble summary",
    ),
    label="tab:ensemble_combined",
    position_float="centering",
    convert_css=True,
)


#%%

scores = pd.read_pickle(PATH / "Data" / "Simulated" / "scores.pkl")
scores_percent = pd.read_pickle(PATH / "Data" / "Simulated" / "scores_percent.pkl")

is_ensemble = scores.index.get_level_values(1) == "Ensembles"
combined = scores_percent.copy()
combined.loc[is_ensemble] = scores.loc[is_ensemble]

# --- Styling / LaTeX ---

def pct_fmt(x):
    # Handle NaNs gracefully for LaTeX
    return f"{x * 100:.1f}\\%" if pd.notna(x) else ""

def num_fmt(x):
    return f"{x:.2f}" if pd.notna(x) else ""

styler = (
    combined.style
    .format(num_fmt, subset=idx[is_ensemble, :], escape="latex")
    .format(pct_fmt, subset=idx[~is_ensemble, :], escape="latex")
    .highlight_between(
        left=0, right=1, inclusive="neither",
        subset=idx[~is_ensemble, :],
        props="font-weight:bold;"
    )
)
styler.to_latex(
    PATH / "Results" / "Tables" / "scores_simulated.tex",
    hrules=True,
    clines="all;data",
    position="htb",
    caption=(
        "Model Scores (Ensemble in Original Units; Other Models as Percent of Ensemble)",
        "Ensemble summary",
    ),
    label="tab:ensemble_combined",
    position_float="centering",
    convert_css=True,
)