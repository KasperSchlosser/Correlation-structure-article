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


def _suffix(kind: str) -> str:
    return "" if kind == "Real" else "_sim"


def _graphs_dir() -> Path:
    return PATH / "Results" / "Graphs"


def _tables_dir() -> Path:
    return PATH / "Results" / "Tables"


def _data_dir(kind: str) -> Path:
    return PATH / "Data" / kind


def _models_dir(kind: str) -> Path:
    return PATH / "Models" / kind / "sarima"


def load_dataset(kind: str):
    base = _data_dir(kind)
    data = {
        "ens": pd.read_pickle(base / "ensembles.pkl"),
        "obs": pd.read_pickle(base / "observations.pkl"),
        "train": pd.read_pickle(base / "modeltraining.pkl"),
        "marginal_quantiles": pd.read_pickle(base / "marginal_quantiles.pkl"),
        "pseudo": pd.read_pickle(base / "pseudoresidual.pkl"),
        "residuals": pd.read_pickle(base / "residuals_normal.pkl"),
        "sims": pd.read_pickle(base / "simulations_plot.pkl"),
        "scores": pd.read_pickle(base / "scores.pkl"),
        "scores_percent": pd.read_pickle(base / "scores_percent.pkl"),
    }
    return data


def plot_raw_data(data, kind: str):
    ens, obs = data["ens"], data["obs"]
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), squeeze=True)
    axes = axes.ravel()
    for i, zone in enumerate(zones):
        tmp_ens = ens.loc[zone].loc[example_date[0] : example_date[1]]
        tmp_obs = obs.loc[zone].loc[example_date[0] : example_date[1]]
        axes[i].plot(tmp_ens.index, tmp_ens, color="black", alpha=0.05, label="Ensemble")
        axes[i].scatter(tmp_obs.index, tmp_obs, s=1, label="Actual Production")
        axes[i].set_title(zone)
        axes[i].tick_params(axis="x", labelrotation=20)

    axes[2].set_xlim(*example_date)
    fig.supxlabel("Date")
    fig.supylabel("MWh")
    fig.set_layout_engine("constrained")
    fig.savefig(_graphs_dir() / f"data_example{_suffix(kind)}.pdf")

    train_size = 1 - 0.46735
    timepoints = obs.index.get_level_values(1).unique()
    print(f"[{kind}] Period:{timepoints.min()} {timepoints.max()} Total data: {len(timepoints)} hours")
    print(f"[{kind}] split at {train_size:.2%} -> {int(len(timepoints)*train_size)} hours train, {int(len(timepoints)*(1-train_size))} hours test")
    print(f"[{kind}] Train period: {timepoints.min()} to {timepoints[int(len(timepoints)*train_size)-1]}")


def summary_tables(data, kind: str):
    ens, obs = data["ens"], data["obs"]
    suffix = _suffix(kind)

    tmp = obs.unstack(level=0).describe().drop("count")
    tmp.style.format(precision=2).format_index(escape="latex").to_latex(
        _tables_dir() / f"observations_summary{suffix}.tex",
        position="htb",
        hrules=True,
        caption=("Summary Statistics for the observed power productions", "Observation summary"),
        label=f"tab:observation{suffix}",
        position_float="centering",
    )

    tmp = ens.unstack(level=0).stack(level=0, future_stack=True).describe().drop("count")
    tmp.style.format(precision=2).format_index(escape="latex").to_latex(
        _tables_dir() / f"ensembles_summary{suffix}.tex",
        position="htb",
        hrules=True,
        caption=("Summary Statistics for the Ensembles", "Ensemble summary"),
        label=f"tab:ensemble{suffix}",
        position_float="centering",
    )


def plot_training_loss(data, kind: str):
    train = data["train"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), squeeze=True)
    axes = axes.ravel()
    for i, zone in enumerate(zones):
        for model, df in train.loc[zone].groupby(level=0):
            line = axes[i].plot(df["Train loss"].values, label=model)
            axes[i].plot(df["Validation loss"].values, ls="--", color=line[0].get_color())
        axes[i].set_title(zone)
        axes[i].set_ylim([np.nanmin(train.loc[zone].values), train.loc[zone].xs(25, level=1).values.max()])

    axes[1].plot(0, 0, color="black", label="Train Loss")
    axes[1].plot(0, 0, color="black", ls="--", label="Validation Loss")
    axes[1].legend()

    fig.supxlabel("Epoch")
    fig.supylabel("Quantile Loss")
    fig.set_layout_engine("constrained")
    fig.savefig(_graphs_dir() / f"loss_curves{_suffix(kind)}.pdf")


def plot_marginals(data, kind: str):
    marginal_quantiles = data["marginal_quantiles"]
    obs = data["obs"]
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), squeeze=True)
    axes = axes.ravel()
    for i, zone in enumerate(zones):
        q = marginal_quantiles.loc[zone, "Latent"]
        x = q.index
        line = axes[i].plot(x, q["0.50"])
        for j in range(1, len(marginal_quantiles.columns) // 2):
            axes[i].fill_between(x, q.iloc[:, j], q.iloc[:, -j - 1], color=line[0].get_color(), alpha=0.2)
        axes[i].scatter(obs.loc[zone].index, obs.loc[zone], s=3, color="black")
        axes[i].set_title(zone)
        axes[i].tick_params(axis="x", labelrotation=20)

    axes[2].set_xlim(*example_date)
    fig.supxlabel("Date")
    fig.supylabel("MWh")
    fig.set_layout_engine("constrained")
    fig.savefig(_graphs_dir() / f"marginal_example{_suffix(kind)}.pdf")


def plot_pseudo_residual(data, kind: str):
    marginal_quantiles = data["marginal_quantiles"]
    pseudo = data["pseudo"]
    fig, ax = plt.subplots(figsize=(8, 6))
    pr = pseudo.loc["DK1-onshore", "Latent"]
    line = ax.axhline(0, label="Median")
    for i in range(1, len(marginal_quantiles.columns) // 2):
        q1 = stats.norm.ppf(float(marginal_quantiles.columns[i]))
        q2 = stats.norm.ppf(float(marginal_quantiles.columns[-i - 1]))
        ax.axhspan(q1, q2, color=line.get_color(), alpha=0.2)
    ax.scatter(pr.index, pr, s=3, color="black", label="Actual production")
    ax.set_title("DK1-onshore")
    ax.set_xlabel("Date")
    ax.set_ylabel("MWh")
    ax.tick_params(axis="x", labelrotation=20)
    ax.set_xlim(*example_date)
    fig.set_layout_engine("constrained")
    fig.savefig(_graphs_dir() / f"pseudoresidual{_suffix(kind)}.pdf")


def parameters_table(kind: str):
    params = pd.DataFrame(
        columns=pd.MultiIndex.from_product([zones, ["Latent", "Simple"]]),
        index=[r"$\\theta_1$", r"$\\phi_1$", r"$\\Theta_1$", r"$\\sigma^2$"],
        dtype=np.float64,
    )
    for f in _models_dir(kind).glob("*"):
        zone, model = f.stem.split("_")
        data = np.load(f)
        params[zone, model] = data

    params.style.format(precision=2).to_latex(
        _tables_dir() / f"parameters{_suffix(kind)}.tex",
        hrules=True,
        clines="all;data",
    )


def plot_acf_pacf(data, kind: str):
    pseudo = data["pseudo"]
    residuals = data["residuals"]
    width = 0.1
    nlags = 48
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6), squeeze=True)
    for i, zone in enumerate(zones):
        X = np.arange(1,49) + 0.25 * (i - 2)
        acf_before = sm.tsa.acf(pseudo.loc[zone, "Latent"], nlags=nlags)
        pacf_before = sm.tsa.pacf(pseudo.loc[zone, "Latent"], nlags=nlags)
        acf_after = sm.tsa.acf(residuals.loc[zone, "Latent"], nlags=nlags)
        pacf_after = sm.tsa.pacf(residuals.loc[zone, "Latent"], nlags=nlags)
        axes[0, 0].bar(X, acf_before[1:], width=width, label=zone)
        axes[0, 1].bar(X, pacf_before[1:], width=width, label=zone)
        axes[1, 0].bar(X, acf_after[1:], width=width, label=zone)
        axes[1, 1].bar(X, pacf_after[1:], width=width, label=zone)

    fig.supxlabel("Lag")
    axes[0, 0].set_title("ACF")
    axes[0, 1].set_title("PACF")
    axes[0, 0].set_ylabel("Before")
    axes[1, 0].set_ylabel("after")
    axes[0, 1].legend()
    fig.set_layout_engine("constrained")
    fig.savefig(_graphs_dir() / f"acf_pacf{_suffix(kind)}.pdf")


def plot_autocorrelation(data, kind: str):
    sims = data["sims"]
    obs = data["obs"]
    marginal_quantiles = data["marginal_quantiles"]
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), squeeze=True)
    axes = axes.ravel()
    for i, zone in enumerate(zones):
        q = sims.loc[zone, "Latent"]
        x = q.index
        line = axes[i].plot(x, q["0.50"])
        for j in range(1, len(marginal_quantiles.columns) // 2):
            axes[i].fill_between(x, q.iloc[:, j], q.iloc[:, -j - 1], color=line[0].get_color(), alpha=0.2)
        axes[i].scatter(obs.loc[zone].index, obs.loc[zone], s=3, color="black")
        axes[i].set_title(zone)
        axes[i].tick_params(axis="x", labelrotation=20)

    axes[2].set_xlim(*example_date)
    fig.supxlabel("Date")
    fig.supylabel("MWh")
    fig.set_layout_engine("constrained")
    fig.savefig(_graphs_dir() / f"corrolation_example{_suffix(kind)}.pdf")


def scores_table(data, kind: str):
    scores = data["scores"]
    scores_percent = data["scores_percent"]
    is_ensemble = scores.index.get_level_values(1) == "Ensembles"
    combined = scores_percent.copy()
    combined.loc[is_ensemble] = scores.loc[is_ensemble]

    def pct_fmt(x):
        return f"{x * 100:.1f}\\%" if pd.notna(x) else ""

    def num_fmt(x):
        return f"{x:.2f}" if pd.notna(x) else ""

    styler = (
        combined.style
        .format(num_fmt, subset=idx[is_ensemble, :], escape="latex")
        .format(pct_fmt, subset=idx[~is_ensemble, :], escape="latex")
        .highlight_between(left=0, right=1, inclusive="neither", subset=idx[~is_ensemble, :], props="font-weight:bold;")
    )

    filename = "scores.tex" if kind == "Real" else "scores_simulated.tex"
    styler.to_latex(
        _tables_dir() / filename,
        hrules=True,
        clines="all;data",
        position="htb",
        caption=("Model Scores (Ensemble in Original Units; Other Models as Percent of Ensemble)", "Ensemble summary"),
        label="tab:ensemble_combined",
        position_float="centering",
        convert_css=True,
    )


def generate(kind: str):
    data = load_dataset(kind)
    plot_raw_data(data, kind)
    summary_tables(data, kind)
    plot_training_loss(data, kind)
    plot_marginals(data, kind)
    plot_pseudo_residual(data, kind)
    parameters_table(kind)
    plot_acf_pacf(data, kind)
    plot_autocorrelation(data, kind)
    scores_table(data, kind)


if __name__ == "__main__":
    for kind in ["Real", "Simulated"]:
        generate(kind)

