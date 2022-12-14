"""
Search for optimal aggregation parameters using Bayesian optimization.

Free parameters we consider:

  - Extremization factor, d
  - Forecast-specific weights
    - Forecast half-life
  - Forecaster-specific weights
    - Metaculus reputation
    - Average update frequency
    - Average update magnitude
"""

import pandas as pd

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from utils import score_preds, log_score
from aggregation import get_aggregate, neyman_agg, neyman_opt

bdf = pd.read_csv("data/bdf.csv")


def lscore(d, hl, rep_weight, mag_weight, freq_weight):
    return -score_preds(
        get_aggregate(
            bdf,
            neyman_agg,
            {"get_d": lambda _: d,
             "half_life": hl,
             "rep_weight": rep_weight,
             "mag_weight": mag_weight,
             "freq_weight": freq_weight,
             "relative_t": True,
             "col_name": "np_test"}),
        ["np_test"],
        log_score).mean()[0]


pbounds = {
    'd'           : (1, 3**.5),
    'hl'          : (.01, 2),
    'rep_weight'  : (.01, 2),
    'mag_weight'  : (.01, 2),
    'freq_weight' : (.01, 2),
    }

optimizer = BayesianOptimization(
    f=lscore,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

logger = JSONLogger(path="data/opt_logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)


optimizer.maximize(
    init_points=5,
    n_iter=100,
)
