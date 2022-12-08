import numpy as np
import matplotlib as mpl

# commented out unused import
# import pandas as pd
# commented out unused import
# import seaborn as sns

from utils import get_features_of_latest_forecasts, log_score, score_preds
from aggregation import neyman_agg
from utils import load_data, get_bdf

plt = mpl.pyplot
plt.rc('font', size=22)
plt.rc('figure', figsize=(1.5*16, 1.5*9))


# loads binary data only
data = load_data(continuous=False)

# creates dataframe of binary predictions enriched with useful question info
# loading only first 5k rows because the analysis below will crash my laptop
# with a lager dataset
bdf = get_bdf(data, nrows=5000)

# creates dictionary of "prediction histories" for every question at every
# point in time
phs = get_features_of_latest_forecasts(bdf)

# calculate Neyman aggregation for every question at every time
bdf['np'] = np.array([neyman_agg(ph) for ph in phs]).flatten()

# Score every aggregate -- neyman > metaculus > community
scores = score_preds(bdf, ['cp', 'mp', 'np'], log_score)
