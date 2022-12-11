import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

import scipy


def enhance_df(main_df, new_df):
    enhanced_df = main_df.merge(
        new_df, how="left", left_on="question_id", right_on="question_id_p")
    enhanced_df = enhanced_df[['question_id', 'title', 'title_short',
                              'description', 'resolution', 'resolution_comment',
                               'categories', 'mp_p', 'cp_p']].\
        drop_duplicates(subset=['title'])
    enhanced_df = enhanced_df[enhanced_df.resolution_comment == "resolved"]
    enhanced_df = enhanced_df[enhanced_df.mp_p.notnull()]
    # Create new columns with factors we care about; text length and accuracy
    enhanced_df['description_len'] = enhanced_df['description'].str.len()
    # the below line normalizes the length for more intuitive regressions
    enhanced_df['description_len'] = np.divide(
        enhanced_df['description_len'], enhanced_df['description_len'].max())
    enhanced_df['cp_accuracy'] = np.exp(enhanced_df['resolution'].subtract(
        enhanced_df['cp_p'], 2))
    enhanced_df['mp_accuracy'] = np.exp(enhanced_df['resolution'].subtract(
        enhanced_df['mp_p']))
    return enhanced_df


def standard_scatter_and_regression(x_axis, x_distribution, y_axis, chart_title):
    fig = mp.figure()
    m_1, b_1 = np.polyfit(x_distribution, y_axis, 1)
    mp.scatter(x_axis, y_axis)
    mp.plot(x_axis, m_1*x_axis+b_1)
    mp.text(.832, 2.149, str(m_1))
    fig.savefig("figures/" + chart_title + ".png")
    mp.close(fig)


def calc_corr_length_pred(df, return_columns):
    cat_correlations = pd.DataFrame(columns=return_columns)
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
