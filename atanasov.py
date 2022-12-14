"""
Check whether Metaculus data supports the meta-analytic finding in [Atanasov & Himmelstein (2022)](https://psyarxiv.com/rm49a/) that frequency and magnitude of updates predict forecasting accuracy, with better forecasters making more, smaller updates.

If this bears out, frequency and magnitude of updates can be used as weights for aggregation. 
"""

import pandas as pd
from utils import get_bdf, score_preds, log_score

bdf = get_bdf()
assert bdf.t.is_monotonic_increasing


def replacena(value, new):
    return value if not pd.isna(value) else new


uids = bdf.user_id.unique().tolist()
qids = bdf.question_id.unique().tolist()

mag = pd.DataFrame(0, index=uids, columns=qids)
freq = pd.DataFrame(0, index=uids, columns=qids)
count = pd.DataFrame(0, index=uids, columns=qids)

bdf['mag_at_t'] = 0
bdf['freq_at_t'] = 0

for k, row in bdf.iterrows():
    if k % (bdf.shape[0] // 10) == 0:
        print(f"{1e2*(k+1)/bdf.shape[0]:.2f}% done")

    uid = row.user_id
    qid = row.question_id

    count.loc[uid, qid] += 1
    mag.loc[uid, qid] += abs(mag.loc[uid, qid] - row.lo) / count.loc[uid, qid]
    freq.loc[uid, qid] += abs(freq.loc[uid, qid] - row.relative_t) / count.loc[uid, qid]

    # Discard (user, question) pairs with only one prediction (i.e. no updates)
    bdf.loc[k, 'mag_at_t'] = replacena((mag.mask(count <= 1)).mean(axis=1, skipna=True)[uid], 0)
    bdf.loc[k, 'freq_at_t'] = replacena((freq.mask(count <= 1)).mean(axis=1, skipna=True)[uid], 0)

# Standardize variables to be used as aggregation weights
bdf[['nrep', 'nmag', 'nfreq']] = (lambda d: (d-d.mean())/d.std())(bdf[['reputation_at_t', 'mag_at_t', 'freq_at_t']])

bdf['ls'] = score_preds(bdf, ['prediction'], log_score)
print(bdf[['nmag', 'nfreq', 'ls']].corr())
