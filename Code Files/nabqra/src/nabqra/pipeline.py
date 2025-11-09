# %%
import pandas as pd
import numpy as np

from pandas import IndexSlice as idx


class pipeline:

    def __init__(self, correlation_model, quantile_model=None):

        self.quantile_model = quantile_model
        self.correlation_model = correlation_model

    def run(self, estimated_quantiles, observations, train_idx, **kwargs):

        res_cols = pd.MultiIndex.from_product(
            (["Original", "CDF", "Normal"], ["Observation", "Estimate", "Upper Interval", "Lower Interval"]),
            names=["Space", "Result Type"],
        )
        sim_cols = pd.MultiIndex.from_product(
            (["Original", "CDF", "Normal"], ["Simulation" + str(x + 1) for x in range(self.correlation_model.n_sim)]),
            names=["Space", "Simulation"],
        )

        res = pd.DataFrame(index=observations.index, columns=res_cols, dtype=np.float64)
        sim = pd.DataFrame(index=observations.index, columns=sim_cols, dtype=np.float64)

        res["Original", "Observation"] = observations

        if self.quantile_model is not None:

            tmp = self.quantile_model.transform(estimated_quantiles.values, res["Original", "Observation"].values)
            res.loc[:, idx["Normal", "Observation"]] = tmp[0]
            res.loc[:, idx["CDF", "Observation"]] = tmp[1]

            tmp = self.correlation_model.transform(res.loc[:, idx["Normal", "Observation"]], train_idx, **kwargs)
            res.loc[idx[:], idx["Normal", ["Estimate", "Upper Interval", "Lower Interval"]]] = tmp[0].values
            sim.loc[idx[:], idx["Normal", :]] = tmp[1].values

            tmp = self.quantile_model.back_transform(
                estimated_quantiles.values,
                res.loc[idx[:], idx["Normal", ["Estimate", "Upper Interval", "Lower Interval"]]].values,
            )
            res.loc[idx[:], idx["Original", ["Estimate", "Upper Interval", "Lower Interval"]]] = tmp[0]
            res.loc[idx[:], idx["CDF", ["Estimate", "Upper Interval", "Lower Interval"]]] = tmp[1]

            tmp = self.quantile_model.back_transform(estimated_quantiles.values, sim.loc[:, idx["Normal", :]].values)
            sim.loc[idx[:], idx["Original", :]] = tmp[0]
            sim.loc[idx[:], idx["CDF", :]] = tmp[1]

        else:

            tmp = self.correlation_model.transform(res.loc[:, idx["Original", "Observation"]], **kwargs)
            res.loc[idx[:], idx["Original", ["Estimate", "Upper Interval", "Lower Interval"]]] = tmp[0].values
            sim.loc[idx[:], idx["Original", :]] = tmp[1].values

        return res, sim
