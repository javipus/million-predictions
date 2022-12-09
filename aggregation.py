import numpy as np


def neyman_agg(prediction_history):

    ps = prediction_history.prediction
    # print(ps)
    ts = prediction_history.question_lifetime_portion_elapsed
    ts -= ts.max()

    reps = prediction_history.reputation_at_t
    reps -= reps.max()

    ws = np.exp(ts + reps)
    ws /= np.sum(ws)
    assert abs(np.sum(ws) - 1) < 1e-6

    # commenting this line out as it's not being used in the script
    # wps = ws*ps
    # print(type(prediction_history))
    # print(prediction_history)
    # print(type(ps))
    n = ps.shape[0]
    k = (n*(3*n**2-3*n+1)**(.5)-2) / (n**2-n-1)

    # This is a new attempt at the calculation logic
    # using only vectorized functions using C and
    # should be significantly faster than
    # the previous implementation. Untested.
    native_l2p = np.divide(np.power(2, ps), 1 + np.power(2, ps))
    native_p2l = np.log2(np.divide(ps, 1 - ps))
    return native_l2p(k*sum(ws*native_p2l(ps)))
