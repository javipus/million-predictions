import pandas as pd

from utils import log_score, score_preds
from aggregation import neyman_agg
from utils import get_bdf, load_data  # scores_path

try:
    # Load scores from file if available
    # scores = pd.read_csv(scores_path)
    raise FileNotFoundError
except FileNotFoundError:
    # Calculate scores if not available
    # Load binary prediction data
    data = load_data(continuous=False)
    bdf = get_bdf(data)

    # We're gonna aggregate predictions using our formula question by question
    qids = bdf['question_id'].unique()
    qs = []
    for qid in qids:
        qs.append(neyman_agg(
            bdf[bdf.question_id == qid]))

    # Replace binary predictions dataframe with another one
    # containing our aggregate
    bdf = pd.concat(qs, ignore_index=True)

    # Score every aggregate -- hopefully neyman > metaculus > community
    scores = score_preds(bdf, ['cp', 'mp', 'np'], log_score)

print(scores.describe())
ax = scores.boxplot()
ax.set_yscale('log')
ax.get_figure().show()
