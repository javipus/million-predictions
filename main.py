import numpy as np
# note, pyplot must be specifically imported as below
import matplotlib.pyplot as plt
# commented out unused imports
# import pandas as pd
# import seaborn as sns

from utils import get_features_of_latest_forecasts, meta_df_prediction_data, log_score, score_preds
from aggregation import neyman_agg
from utils import load_data, get_bdf
from datetime import datetime

startTime = datetime.now()

plt.rc('font', size=22)
plt.rc('figure', figsize=(1.5*16, 1.5*9))


# loads binary data only
data = load_data(continuous=False)

# creates dataframe of binary predictions enriched with useful question info
# loading only first 5k rows because the analysis below will crash my laptop
# with a lager dataset
bdf = get_bdf(data)  # , nrows=5000)

# creates dictionary of "prediction histories" for every question at every
# point in time
# phs = get_features_of_latest_forecasts(bdf)
phs = meta_df_prediction_data(bdf)

print("phs data type is: " + str(phs.dtypes))
# calculate Neyman aggregation for every question at every time
bdf['np'] = np.array([neyman_agg(ph) for ph in phs]).flatten()
# bdf['np'] = phs.apply(neyman_agg).values  # , axis=1)


# # Score every aggregate -- neyman > metaculus > community
# scores = score_preds(bdf, ['cp', 'mp', 'np'], log_score)

# # Print the aggregate rankings
# for col in scores:
#     print("The mean " + col + " is: " + str(scores[col].mean()))

# Report runtime
print("This script ran in: " + str(datetime.now() - startTime))
