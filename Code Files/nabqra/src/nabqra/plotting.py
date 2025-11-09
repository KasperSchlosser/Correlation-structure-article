import matplotlib.pyplot as plt

import numpy as np
import statsmodels.api as sm
import scipy.stats as stats


def _plot_correlation_function(values, ax, alpha, nlags, title, ylabel, func):
    corr, confint = func(values, alpha=alpha, nlags=nlags)
    lags = list(range(len(corr)))
    bars = ax.bar(lags, corr)

    lower_confidence = confint[:, 0] - corr
    upper_confidence = confint[:, 1] - corr
    ax.fill_between(lags, lower_confidence, upper_confidence, color=bars[0].get_facecolor(), alpha=0.3)

    ax.set_xlabel("Lag")
    ax.set_ylabel(ylabel)


def plot_acf(values, ax, alpha=0.05, nlags=48):
    _plot_correlation_function(values, ax, alpha, nlags, "ACF Plot", "ACF", sm.tsa.acf)


def plot_pacf(values, ax, alpha=0.05, nlags=48):
    _plot_correlation_function(values, ax, alpha, nlags, "PACF Plot", "PACF", sm.tsa.pacf)


def plot_uniform_histogram(uniform_residuals, ax):
    ax.hist(uniform_residuals, density=True, bins="auto")
    ax.hlines(1, 0, 1, colors="black", linestyles="dashed")
    ax.set_xlabel("u")
    ax.set_ylabel("Density")


def plot_normal_histogram(normal_residuals, ax):
    xgrid = np.linspace(min(normal_residuals), max(normal_residuals), 1000)
    kde = stats.gaussian_kde(normal_residuals)

    ax.hist(normal_residuals, density=True, bins="auto")
    ax.plot(xgrid, kde(xgrid), label="KDE")
    ax.plot(xgrid, stats.norm.pdf(xgrid), color="black", linestyle="dashed", label="Standard Normal PDF")

    ax.set_xlabel("z")
    ax.set_ylabel("Density")


def plot_outliers(normal_residuals, index, ax, alpha=0.05):

    alpha_bonferroni = alpha / len(index)
    limits = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    limits_bonferroni = stats.norm.ppf([alpha_bonferroni / 2, 1 - alpha_bonferroni / 2])

    ax.scatter(index, normal_residuals)
    ax.hlines(0, *index[[0, -1]], color="black")
    ax.hlines(limits, *index[[0, -1]], color="green", label=f"{1-alpha:.0%} Interval")
    ax.hlines(limits_bonferroni, *index[[0, -1]], color="red", label=f"{1-alpha:.0%} Interval - Bonferroni corrected")

    ax.set_xlabel("Date")
    ax.set_ylabel("z")


def plot_qq(normal_residuals, ax):
    theo, obs = stats.probplot(normal_residuals, dist="norm", fit=False)
    ax.scatter(theo, obs)
    ax.axline((0, 0), slope=1, color="black")
    ax.set_xlabel("Theoretical Quantile")
    ax.set_ylabel("Observed Quantile")


def diagnostic_plots(normal_residuals, index, save_path=None, individual=False, closefig=False):

    if save_path:
        save_dir = save_path.parent
        base_name = save_path.stem

    uniform_residuals = stats.norm.cdf(normal_residuals)

    plots = ("acf", "pacf", "uniform_histogram", "normal_histogram", "outliers", "qq")

    if individual:
        figs, axes = zip(*[plt.subplots() for _ in plots])

    else:
        fig, axes = plt.subplots(3, 2, figsize=(8, 12))
        axes = axes.flatten()
        figs = [fig]

    plot_acf(normal_residuals, axes[0])
    plot_pacf(normal_residuals, axes[1])
    plot_uniform_histogram(uniform_residuals, axes[2])
    plot_normal_histogram(normal_residuals, axes[3])
    plot_outliers(normal_residuals, index, axes[4])
    plot_qq(normal_residuals, axes[5])

    if save_path:
        if individual:
            for plot, fig in zip(plots, figs):
                fig.savefig(save_dir / (f"{base_name}_{plot}"))
        else:
            plt.savefig(save_dir / f"{base_name}")

    if closefig:
        if individual:
            for plot, fig in zip(plots, figs):
                plt.close(fig)
        else:
            plt.savefig(save_dir / f"{base_name}")
            plt.close(fig)

    return figs, axes


def multi_y_plot(
    X, Ys, ax=None, labels=None, ylims=None, offset=0.05, hide_original=True, color_cycler=None, label_pos="Legend"
):

    if ax is None:
        fig, ax = plt.subplots()

    else:
        fig = ax.get_figure()

    if labels is None:
        labels = [None for _ in range(len(Ys))]
    if ylims is None:
        ylims = [None for _ in range(len(Ys))]
    if color_cycler is None:
        color_cycler = plt.rcParams["axes.prop_cycle"]()

    if hide_original:
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.grid(False)

    offsets = [-offset * x for x in range(len(Ys))]

    lines = []
    for y, label, ylim, offset, color in zip(Ys, labels, ylims, offsets, color_cycler):

        color = color["color"]

        twin = ax.twinx()

        line = twin.plot(X, y, color=color)
        lines.append(line[0])

        twin.spines["left"].set_visible(True)
        twin.spines["left"].set_position(("axes", offset))
        twin.spines["left"].set_color(color)

        twin.yaxis.set_label_position("left")
        twin.yaxis.set_ticks_position("left")
        twin.tick_params(axis="y", which="both", colors=color, labelrotation=90)

        twin.set_ylim(ylim)

        twin.grid(False)

        if label_pos == "Axis":
            twin.set_ylabel(label, labelpad=0, color=color)
        else:
            twin.set_ylabel("", labelpad=0, color=color)

    if label_pos == "Legend":
        ax.legend(lines, labels)

    return fig, ax


def band_plot(x, y_mid, y1, y2=0, ax=None, color=None, alpha=0.3, label=None, band_label=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    p = ax.plot(x, y_mid, label=label, color=color)
    color = p[0].get_color()

    ax.fill_between(x, y1, y2, color=color, alpha=alpha, label=band_label)

    return fig, ax
