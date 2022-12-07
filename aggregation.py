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

  wps = ws*ps

  n = len(ps)
  k = (n*(3*n**2-3*n+1)**(.5)-2) / (n**2-n-1)

  return np.vectorize(l2p)(k*sum(ws*np.vectorize(p2l)(ps)))
