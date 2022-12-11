import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from sklearn.linear_model import LinearRegression

from utils import load_data  # scores_path,

# make a dataframe for question data only
q_only_data = load_data(continuous=False, question_only=True)
q_only = pd.DataFrame(q_only_data["questions"]["binary"])

# get a dataframe with all binary question/prediction data.
p_data = load_data(continuous=False, question_only=False)
p_only = pd.DataFrame(p_data["predictions"]["binary"]).add_suffix("_p")

# These lines check the size and datatypes of q_only
# print(q_only.info())
# print(q_only.tail())

# produce an enhanced question dataframe that includes mp and cp
q_enhanced = q_only.merge(
    p_only, how="left", left_on="question_id", right_on="question_id_p")
q_enhanced = q_enhanced[['question_id', 'title', 'title_short',
                         'description', 'resolution', 'resolution_comment',
                         'categories', 'mp_p', 'cp_p']].\
    drop_duplicates(subset=['title'])
q_enhanced = q_enhanced[q_enhanced.resolution_comment == "resolved"]
q_enhanced = q_enhanced[q_enhanced.mp_p.notnull()]

# Create new columns with factors we care abuot; text length and accuracy
q_enhanced['description_len'] = q_enhanced['description'].str.len()
# the below line normalizes the length for more intuitive regressions
q_enhanced['description_len'] = np.divide(
    q_enhanced['description_len'], q_enhanced['description_len'].max())
q_enhanced['cp_accuracy'] = np.exp(q_enhanced['resolution'].subtract(
    q_enhanced['cp_p'], 2))
q_enhanced['mp_accuracy'] = np.exp(q_enhanced['resolution'].subtract(
    q_enhanced['mp_p']))

# get a sense for distribution of data
# q_enhanced.plot(x="description_len", y=[
# "cp_accuracy", "mp_accuracy"], kind="bar", figsize=(9, 8))
# mp.show()

# plot linear regressions against our points!
# perform linear regression
# print(type(]))
# print(type(q_enhanced['cp_accuracy']))

var_len = q_enhanced['description_len'].values.reshape(-1, 1)
var_cp_acc = q_enhanced['cp_accuracy'].values.reshape(-1, 1)
var_mp_acc = q_enhanced['mp_accuracy'].values.reshape(-1, 1)

print(len(var_len))
print(len(var_cp_acc))
print(len(var_mp_acc))

# # build linear regression with cp
# linear_regressor_cp = LinearRegression().fit(
#     var_len, var_cp_acc)
# cp_pred = linear_regressor_cp.predict(
# var_len)
#
# build linear regression with mp
# linear_regressor_mp = LinearRegression()
# linear_regressor_mp.fit(
#     var_len, var_mp_acc)
# mp_pred = linear_regressor_mp.predict(
#     var_mp_acc)  # make predictions

m_1, b_1 = np.polyfit(q_enhanced['description_len'], var_cp_acc, 1)
m_2, b_2 = np.polyfit(q_enhanced['description_len'], var_mp_acc, 1)

fig1 = mp.figure()
mp.scatter(var_len, var_cp_acc)
mp.plot(var_len, m_1*var_len+b_1)
mp.text(.832, 2.149, str(m_1))
mp.show()
fig1.savefig("figures/len_vs_cp.png")

fig2 = mp.figure()
mp.scatter(var_len, var_mp_acc)
mp.plot(var_len, m_2*var_len+b_2)
mp.text(.832, 2.149, str(m_2))
mp.show()
fig2.savefig("figures/len_vs_mp.png")

# mp.plot(var_len, var_cp_acc)
#     var_cp_acc, var_mp_acc]
# mp.scatter(var_len, [
#     var_cp_acc, var_mp_acc])
# mp.plot(var_len, [
#     var_cp_acc, var_mp_acc],  # q_enhanced['mp_accuracy'
# color=['red', 'blue'])
mp.show()
