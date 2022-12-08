from datetime import datetime
from pathlib import Path
from functools import reduce
from math import log2, log
import pandas as pd
# unused import
# import numpy as np

cwd = Path(".").absolute()
data_path = cwd / "data"

# Probability stuff #

# need to fix l2p and p2l


def o2p(o): return o/(1+o)
def p2o(p): return p/(1-p)
def p2l(p): return log2(p2o(p))
# variable name edited from "l" to "lo" to avoid confusion with the number 1
def l2p(lo): return o2p(2**lo)
def boost(k, p): return o2p(k*p2o(p))
def agg_odds(ps, k=1): return o2p(
    (reduce(lambda x, y: x*y, map(p2o, ps))**(1/len(ps)))**k)


def agg_lodds(ps, k=1): return l2p(k*sum(map(p2l, ps)) / len(ps))


def log_score(p, y, base=2):
    return -(y*log(p, base) + (1-y)*log(1-p, base))


def score_preds(bdf, preds, scoring_rule):
    sc = pd.DataFrame([bdf.apply(lambda row: scoring_rule(
        row[pred], row['resolution']), axis=1) for pred in preds]).T
    sc.columns = preds
    return sc

# Data transformation #


def get_bdf(data, nrows=None, transform=None):
    _df = augment_prediction_data(data, 'binary')
    nrows = nrows or _df.shape[0]
    return _df.head(nrows).groupby("question_id"). \
        apply(transform or
              (lambda x: x)).reset_index(drop=True)


def augment_prediction_data(data, _type='binary'):
    # Add data from questions table to predictions, and add useful time
    # features
    questions = data['questions'][_type]
    predictions = data['predictions'][_type]
    questions = questions[questions.resolution_comment == "resolved"]
    questions["duration"] = questions.close_time - questions.publish_time
    question_data = questions[
        [
            "question_id",
            "resolution",
            "created_time",
            "publish_time",
            "close_time",
            "resolve_time",
            "duration",
        ]
    ]
    predictions = predictions.merge(question_data, on="question_id")
    predictions["t"] = predictions["t"].apply(lambda x: x.timestamp())
    predictions["time_to_resolution"] = \
        predictions.resolve_time - predictions.t
    predictions["time_since_publish"] = predictions.t - \
        predictions.publish_time
    predictions["relative_t"] = predictions.time_since_publish / \
        predictions.duration
    return predictions


def get_features_of_latest_forecasts(question_df):

    question_df = question_df.sort_values(by=['t'])

    idxs_of_users_latest_forecasts = get_idxs_of_users_latest_forecasts(
        list(question_df["user_id"]))

    features_of_latest_forecasts = [0, ]*len(question_df.index)

    for i, idxs in enumerate(idxs_of_users_latest_forecasts):
        relevant_df = question_df.iloc[idxs]
        features_of_latest_forecasts[i] = {
            "predictions": list(relevant_df["prediction"]),
            "question_lifetime_portion_elapsed":
            list(relevant_df["relative_t"]),
            "reputations": list(relevant_df["reputation_at_t"]),
            "user_ids": list(relevant_df["user_id"])
        }

    return features_of_latest_forecasts


def add_features_of_latest_forecasts(question_df):
    question_df["features_of_latest_forecasts"] \
        = get_features_of_latest_forecasts(
        question_df)
    return question_df


def get_idxs_of_users_latest_forecasts(user_ids):
    latest_idxs = {}
    idxs_of_users_latest_forecasts = [0, ]*len(user_ids)
    for i in range(len(user_ids)):
        latest_idxs[user_ids[i]] = i
        idxs_of_users_latest_forecasts[i] = [v for k, v in latest_idxs.items()]
    return idxs_of_users_latest_forecasts


def load_data(data_path=data_path, binary=True, continuous=True):
    if binary:
        print("Loading binary questions")
        binary_questions = pd.read_json(
            data_path / "questions-binary-hackathon.json",
            orient="records",
            # This is necessary, otherwise Pandas messes up date conversion.
            convert_dates=False,
        )
        binary_questions.drop(
            ['title', 'title_short', 'description', 'split', 'categories'],
            axis=1)
        binary_questions["question_id"] = \
            binary_questions["question_id"].astype("int16")

        print("Loading binary predictions")
        binary_predictions = pd.read_json(
            data_path / "predictions-binary-hackathon.json",
            orient="records",
        )
        binary_predictions["t"] = binary_predictions["t"].apply(
            datetime.fromtimestamp)
        binary_predictions = binary_predictions.set_index("t", drop=False)
        binary_predictions["user_id"] = binary_predictions["user_id"].astype(
            "int16")
        binary_predictions["question_id"] = \
            binary_predictions["question_id"].astype("int16")
        binary_predictions["prediction"] = \
            binary_predictions["prediction"].astype("float32")
        binary_predictions["reputation_at_t"] = \
            binary_predictions["reputation_at_t"].astype("float32")
        binary_predictions["mp"] = binary_predictions["mp"].astype(
            "float32")
        binary_predictions["cp"] = binary_predictions["cp"].astype(
            "float32")

    if continuous:
        print("Loading continuous questions")
        continuous_questions = pd.read_json(
            data_path / "questions-continuous-hackathon.json",
            orient="records",
            # This is necessary, otherwise Pandas messes up date conversion.
            convert_dates=False,
        )

        print("Loading continuous predictions")
        continuous_predictions = pd.read_parquet(
            data_path / "predictions-continuous-hackathon-v2.parquet"
        )
        continuous_predictions["t"] = continuous_predictions["t"].apply(
            datetime.fromtimestamp)
        continuous_predictions = continuous_predictions.set_index(
            "t", drop=False)
    return {
        "questions": {
            "binary": binary_questions if binary else None,
            "continuous": continuous_questions if continuous else None,
        },
        "predictions": {
            "binary": binary_predictions if binary else None,
            "continuous": continuous_predictions if continuous else None,
        }
    }
