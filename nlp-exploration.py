import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from utils import load_data
from nlp import enhance_df, calc_corr_length_pred, standard_scatter_and_regression
# make a dataframe for question data only
q_only_data = load_data(continuous=False, question_only=True)
q_only = pd.DataFrame(q_only_data["questions"]["binary"])

# get a dataframe with all binary question/prediction data.
p_data = load_data(continuous=False, question_only=False)
p_only = pd.DataFrame(p_data["predictions"]["binary"]).add_suffix("_p")

# produce an enhanced question dataframe that includes key factors of analysis (mp, cp)
q_enhanced = enhance_df(q_only, p_only)

# produce overall regression
# key variables for overall regression
var_len = q_enhanced['description_len'].values.reshape(-1, 1)
var_cp_acc = q_enhanced['cp_accuracy'].values.reshape(-1, 1)
var_mp_acc = q_enhanced['mp_accuracy'].values.reshape(-1, 1)

# overall regressions
standard_scatter_and_regression(
    x_axis=var_len, x_distribution=q_enhanced['description_len'], y_axis=var_cp_acc, chart_title="len_vs_cp")
standard_scatter_and_regression(
    x_axis=var_len, x_distribution=q_enhanced['description_len'], y_axis=var_mp_acc, chart_title="len_vs_mp")

# get more detail for subcategory analysis
q_flatten_categories = pd.get_dummies(
    q_enhanced['categories'].apply(pd.Series).stack()).sum(level=0)
q_enhanced_categories = pd.concat([q_enhanced, q_flatten_categories], axis=1)

# produce dataframe with breakdown of statistical significance and correlation for all categories with >=10 questions.
category_results = calc_corr_length_pred(q_enhanced_categories, return_columns=["category", "var_len", "c_i_cp", "m_cp", "b_cp",
                                                                                "var_cp_acc", "c_i_mp", "m_mp", "b_mp", "var_mp_acc", "t_test_cp",
                                                                                "p_value_cp", "t_test_mp", "p_value_mp"]).squeeze()

large_n_category_results = category_results[category_results["m_mp"] > 0.05]
small_n_category_results = category_results[category_results["p_value_mp"] < 0.05]
print(large_n_category_results.shape)
print(small_n_category_results.shape)

large_n_category_results.sort_values(by="m_mp")
small_n_category_results.sort_values(by="t_test_mp")
avg_slope_large_n = large_n_category_results["m_mp"].mean()
avg_t_statistic_small_n = small_n_category_results["t_test_mp"].mean()
print(avg_slope_large_n)
print(avg_t_statistic_small_n)
print(large_n_category_results["m_mp"])
