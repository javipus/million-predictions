import numpy as np
import pandas as pd
from utils import p2l, l2p

ln2 = np.log(2)


def get_aggregate(bdf, agg_func, agg_func_kwds={}):
    qids = bdf['question_id'].unique()
    qs = []
    for qid in qids:
        qs.append(agg_func(
            bdf[bdf.question_id == qid], **agg_func_kwds))
    return pd.concat(qs, ignore_index=True)


def neyman_agg(q, get_d=None, relative_t=True, half_life=ln2, rep_weight=1, freq_weight=1, mag_weight=1, col_name='np'):
    q = q.sort_values(by=['t']).reset_index(drop=True)
    q['lo'] = q['prediction'].apply(p2l)
    if relative_t:
        dts = q.relative_t.diff().replace(np.nan, 0)
    else:
        dts = q.t.diff().replace(np.nan, 0)

    get_d = get_d or neyman_opt
    ds = [get_d(n) for n in range(1, q.shape[0]+1)]

    los = np.array([])
    ws = np.array([])

    # NB this assumes row index ranges (0, q.shape[0]-1)
    for k, row in q.iterrows():
        if k > 0:
            ws *= np.exp(-dts[k] * ln2 / half_life) if half_life < np.inf else 1
        w_log = rep_weight*row['nrep'] -\
            freq_weight*row['nfreq'] -\
            mag_weight*row['nmag']
        ws = np.append(ws, np.exp(w_log))
        los = np.append(los,
                        ds[k] * np.inner(q.iloc[:k+1, :]['lo'], ws) / sum(ws))

    q[col_name] = l2p(los)

    return q


def neyman_opt(n):
    return n*((3*n**2-3*n+1)**(.5)-2) / (n**2-n-1)
