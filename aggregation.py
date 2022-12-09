import numpy as np
from utils import p2l, l2p

ln2 = np.log(2)


def neyman_agg(q, get_d=None, relative_t=False, half_life=1):
    q = q.sort_values(by=['t']).reset_index(drop=True)
    q['lo'] = q['prediction'].apply(p2l)
    if relative_t:
        dts = q.relative_t.diff().replace(np.nan, 0)
    else:
        dts = q.t.diff().replace(np.nan, 0)

    get_d = get_d or lambda n: (n*(3*n**2-3*n+1)**(.5)-2) / (n**2-n-1)
    ds = [get_d(n) for n in range(1, q.shape[0]+1)]

    los = np.array([])
    ws = np.array([])

    # NB this assumes row index ranges (0, q.shape[0]-1)
    for k, row in q.iterrows():
        if k > 0:
            ws *= np.exp(-dts[k] * ln2 / half_life)
        ws = np.append(ws, np.exp(row['reputation_at_t']))
        los = np.append(los,
                        ds[k] * np.inner(q.iloc[:k+1, :]['lo'], ws) / sum(ws))

    q['np'] = l2p(los)

    return q
