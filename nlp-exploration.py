import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import scipy
from utils import load_data

# make a dataframe for question data only
q_only_data = load_data(continuous=False, question_only=True)
q_only = pd.DataFrame(q_only_data["questions"]["binary"])

# get a dataframe with all binary question/prediction data.
p_data = load_data(continuous=False, question_only=False)
p_only = pd.DataFrame(p_data["predictions"]["binary"]).add_suffix("_p")

# produce an enhanced question dataframe that includes key factors of analysis
q_enhanced = q_only.merge(
    p_only, how="left", left_on="question_id", right_on="question_id_p")
q_enhanced = q_enhanced[['question_id', 'title', 'title_short',
                         'description', 'resolution', 'resolution_comment',
                         'categories', 'mp_p', 'cp_p']].\
    drop_duplicates(subset=['title'])
q_enhanced = q_enhanced[q_enhanced.resolution_comment == "resolved"]
q_enhanced = q_enhanced[q_enhanced.mp_p.notnull()]

# Create new columns with factors we care about; text length and accuracy
q_enhanced['description_len'] = q_enhanced['description'].str.len()
# the below line normalizes the length for more intuitive regressions
q_enhanced['description_len'] = np.divide(
    q_enhanced['description_len'], q_enhanced['description_len'].max())
q_enhanced['cp_accuracy'] = np.exp(q_enhanced['resolution'].subtract(
    q_enhanced['cp_p'], 2))
q_enhanced['mp_accuracy'] = np.exp(q_enhanced['resolution'].subtract(
    q_enhanced['mp_p']))


q_flatten_categories = pd.get_dummies(
    q_enhanced['categories'].apply(pd.Series).stack()).sum(level=0)
q_enhanced_categories = pd.concat([q_enhanced, q_flatten_categories], axis=1)


def calc_corr_length_pred(df):
    cat_correlations = pd.DataFrame(columns=["category", "var_len", "c_i_cp", "m_cp", "b_cp",
                                             "var_cp_acc", "c_i_mp", "m_mp", "b_mp", "var_mp_acc", "t_test_cp", "p_value_cp", "t_test_mp", "p_value_mp"])
    for cat in list(df.iloc[:, 12:]):
        if df[cat].sum() < 10:
            continue
        cat_df = df[df[cat] > 0]
        var_len = cat_df['description_len'].values.reshape(-1, 1)
        var_cp_acc = cat_df['cp_accuracy'].values.reshape(-1, 1)
        var_mp_acc = cat_df['mp_accuracy'].values.reshape(-1, 1)
        if df[cat].sum() > 29:
            # do an analysis based on normal distribution
            m_1, b_1 = np.polyfit(cat_df['description_len'], var_cp_acc, 1)
            m_2, b_2 = np.polyfit(cat_df['description_len'], var_mp_acc, 1)
            c_i_1 = scipy.stats.norm.interval(
                confidence=0.95, loc=np.mean(m_1), scale=scipy.stats.sem(cat_df['description_len']))
            c_i_2 = scipy.stats.norm.interval(
                confidence=0.95, loc=np.mean(m_2), scale=scipy.stats.sem(cat_df['description_len']))
            summary_stats = {"category": cat, "var_len": var_len, "c_i_cp": c_i_1, "m_cp": m_1, "b_cp": b_1,
                             "var_cp_acc": var_cp_acc, "c_i_mp": c_i_2, "m_mp": m_2, "b_mp": b_2, "var_mp_acc": var_mp_acc}

        elif df[cat].sum() > 9:
            # do a t test
            tstat_cp, pvalue_cp = scipy.stats.ttest_ind(
                cat_df['description_len'], var_cp_acc)
            tstat_mp, pvalue_cp = scipy.stats.ttest_ind(
                cat_df['description_len'], var_mp_acc)
            summary_stats = {"category": cat, "var_len": var_len,
                             "var_cp_acc": var_cp_acc, "var_mp_acc": var_mp_acc, "t_test_cp": tstat_cp, "p_value_cp": pvalue_cp, "t_test_mp": tstat_mp, "p_value_mp": pvalue_cp}
        summary_stats_as_series = pd.Series(summary_stats)
        cat_correlations = cat_correlations.append(
            summary_stats_as_series, ignore_index=True)
    return cat_correlations


category_results = calc_corr_length_pred(q_enhanced_categories)

# fig1 = mp.figure()
# mp.scatter(var_len, var_cp_acc)
# mp.plot(var_len, m_1*var_len+b_1)
# mp.text(.832, 2.149, str(m_1))
# mp.show()
# fig1.savefig("figures/len_vs_cp.png")

# fig2 = mp.figure()
# mp.scatter(var_len, var_mp_acc)
# mp.plot(var_len, m_2*var_len+b_2)
# mp.text(.832, 2.149, str(m_2))
# mp.show()
# fig2.savefig("figures/len_vs_mp.png")
