import numpy as np
from utils import p2l, l2p


def neyman_agg(prediction_history):

    ps = np.array(prediction_history['predictions'])

    ts = np.array(
        prediction_history['question_lifetime_portion_elapsed'])
    ts -= max(ts)

    reps = np.array(prediction_history['reputations'])
    reps -= max(reps)

    ws = np.exp(ts + reps)
    ws /= sum(ws)
    assert abs(sum(ws) - 1) < 1e-6

    # commenting this line out as it's not being used in the script
    # wps = ws*ps

    n = len(ps)
    k = (n*(3*n**2-3*n+1)**(.5)-2) / (n**2-n-1)

    # This is a new attempt at the calculation logic using only vectorized functions using C
    # and this should be significantly faster than the previous implementation. Untested.
    native_l2p = np.divide(np.power(2, ps), 1 + np.power(2, ps))
    native_p2l = np.log2(np.divide(ps, 1 - ps))
    return native_l2p(k*sum(ws*native_p2l(ps)))
