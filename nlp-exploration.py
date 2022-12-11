# import numpy as np
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
# print(q_p_data)
# bdf = get_bdf(q_p_data)
# question 1 -- length of description plotted against the accuracy of the
# crowd on each question.
# subquestions include: does it vary for categories? what about subsets
# of a given question, eg, do people who were wrong get "more" wrong re: len?

# print(q_only.info())
# print(q_only.tail())

# we want to get the len() of every description.
# second, we need
# we also want to groupby each category --
# maybe subcategories later.

q_enhanced = q_only.merge(
    p_only, how="left", left_on="question_id", right_on="question_id_p")
q_enhanced = q_enhanced[['question_id', 'title', 'title_short',
                         'description', 'resolution', 'resolution_comment',
                         'categories', 'mp_p', 'cp_p']].\
    drop_duplicates(subset=['title'])
q_enhanced = q_enhanced[q_enhanced.resolution_comment == "resolved"]
q_enhanced = q_enhanced[q_enhanced.mp_p.notnull()]

# print(q_enhanced.info())
# print(q_enhanced.shape)
# # print(q_only.tail())

q_enhanced['description_len'] = q_enhanced['description'].str.len()
q_enhanced['cp_accuracy'] = q_enhanced['resolution'].subtract(
    q_enhanced['cp_p']).abs()
q_enhanced['mp_accuracy'] = q_enhanced['resolution'].subtract(
    q_enhanced['mp_p']).abs()
# print(q_enhanced['accuracy'])

# print(q_enhanced['mp_accuracy'].isna().sum())

# get a sense for distribution of data
# q_enhanced.plot(x="description_len", y=[
# "cp_accuracy", "mp_accuracy"], kind="bar", figsize=(9, 8))
# mp.show()

# plot linear regressions against our points!
linear_regressor_cp = LinearRegression()  # create object for the class
# perform linear regression
# print(type(]))
# print(type(q_enhanced['cp_accuracy']))

var_len = q_enhanced['description_len'].values.reshape(-1, 1)
var_cp_acc = q_enhanced['cp_accuracy'].values.reshape(-1, 1)
var_mp_acc = q_enhanced['mp_accuracy'].values.reshape(-1, 1)

print(len(var_len))
print(len(var_cp_acc))
print(len(var_mp_acc))


linear_regressor_cp.fit(
    var_len, var_cp_acc)
linear_regressor_mp = LinearRegression()
linear_regressor_mp.fit(
    var_len, var_mp_acc)

cp_pred = linear_regressor_cp.predict(
    var_cp_acc)

mp_pred = linear_regressor_mp.predict(
    var_mp_acc)  # make predictions

mp.scatter(var_len, var_cp_acc)
# mp.plot(var_len, var_cp_acc)
#     var_cp_acc, var_mp_acc]
# mp.scatter(var_len, [
#     var_cp_acc, var_mp_acc])
# mp.plot(var_len, [
#     var_cp_acc, var_mp_acc],  # q_enhanced['mp_accuracy'
# color=['red', 'blue'])
mp.show()
