import numpy as np
import pandas as pd
import sklearn


'''
Assume the data set is a PANDAS DataFrame.

01. Frame the problem
Check the objective and existing performance, and then decide:
    a. Supervised? Unsupervised?
    b. Classification? Regression? Sequence tagging?
    c. Batch learning? Online learning?

02. Select a performance measure

03. Check the assumptions

04. Prepare the data
    a. separate the data into train and test sets properly
    b. explore and analyze the train data by visualization and correlations
    c. *combine features
    d. deal with missing values in some features
    e. deal with non-numerical type values in some features
    f. feature scaling
'''

'''------------------------------Separate data-------------------------'''
housing = pd.DataFrame({'a':[0,1,2,3,4], 'b':[5,6,7,8,9]})

'''randomly distribute the test set only if the test set is fixed'''
def split_train_test(data, test_ratio):
    # set the random number generator'seed, so every time generates the same result, ensure the fairness of result comparisons
    np.random.seed(42)
    # return a list that randomly permutes the indices of data set
    shuffled_indices = np.random.permutation(len(data))
    # return an integer indicating the train/test set boundary
    test_set_size = int(len(data) * test_ratio)
    # return a list of indices of test instances
    test_indices = shuffled_indices[:test_set_size]
    # return a list of indices of train instances
    train_indices = shuffled_indices[test_set_size:]
    # test set is the rows of indices in test_indices
    return data.iloc[train_indices], data.iloc[test_indices]

'''above function can be replaced by sklearn's function:'''
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42) # where 'data' is arrays object from any 3rd party source

'''stratified sampling: ensure the test set is representative in the most important ONE feature'''
# convert values in column 'median_income' into certain categories
housing["income_cat"] = np.ceil(housing["a"] / 1.5)
# convert values in column that are < 5 to 5.
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
# create an instance of the imported class
# n_splits: number of re-shuffling & splitting iterations, the result of each iteration is: (train_index_list, test_index_list)
#           if multiple iterations, the results will be concatenated: [(train1, test1), (train2, test2), (train3, test3), ...]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# split the given data according the given label array
for train_index_list, test_index_list in split.split(housing, housing["income_cat"]):
    # the indices for train and test
    strat_train_set = housing.loc[train_index_list]
    strat_test_set = housing.loc[test_index_list]

# if the test set will be updated later, write a hash function to convert
# each instance's unique identifier to decide if the instance should be in test set


'''-------------------------------Visualization------------------------------------------------'''
# chose the graph type, then select the columns/features you'd like to show relations
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)





'''-------------------------------Find correlations between features----------------------------'''
'''Numerical check'''
corr_matrix = housing.corr()
# select a column, and function below displays pairwise correlation of columns to selected column, excluding NA/null values
# default is Pearson's relation, a measure of the linear correlation between two variables X and Y.
# 1 means strong positive correlation while -1 means strong negative correlation and 0 means no linear correlation
corr_matrix["median_house_value"].sort_values(ascending=False)

'''Graph check'''
from pandas.plotting import scatter_matrix
# the list of features you'd like to check relations
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# return a figure, each row represents current feature's relations with others
scatter_matrix(housing[attributes], figsize=(12, 8))




'''------------------------------Combine relative features---------------------------------------'''
'''
Some features alone don't offer much useful information, and also may have low correlations with
target feature; but if combined with other features by some computation, the correlation may improve.

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
'''




'''------------------------------Missing values in features-------------------------------------------------'''
# Get rid of the corresponding instances.
housing.dropna(subset=["total_bedrooms"])
# Get rid of the feature/column.
housing.drop("total_bedrooms", axis=1)
# Set the missing values to some value.
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)

'''sklearn way to fill in missing values'''
from sklearn.preprocessing import Imputer
# define a strategy to deal with missing values when creating an instance
imputer = Imputer(strategy="median")
# get rid of string type features/columns
housing_num = housing.drop("ocean_proximity", axis=1)
# fit the instance with the data, generates a statics in "imputer.statistics_"
imputer.fit(housing_num)
# check the result median column
print(imputer.statistics_)
# fill in and transform the original train data
X = imputer.transform(housing_num)
# convert X back to pandas' DataFrame object
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

'''---------------------------------Non-numerical values in features--------------------------------'''
'''Encode each label/non-numerical value into a unique number, from 0 to len(column)'''
'''Issue: will assume that two nearby values are more similar than two distant values.'''
from sklearn.preprocessing import LabelEncoder
# create an instance of this helper class
encoder = LabelEncoder()
# select the feature/column that has non-numerical values
housing_cat = housing["ocean_proximity"]
# perform encoding procedure; fit_transform() = fit() + transform()
housing_cat_encoded = encoder.fit_transform(housing_cat)
# check the mapping
print(encoder.classes_)

'''Encode each label/non-numerical value into a one-hot-vector'''
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
# returns a SciPy sparse matrix
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
# convert to numpy ndarray
housing_cat_1hot.toarray()


'''Encode from text categories to integer categories, then from integer categories to one-hot vectors in one shot'''
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
# returns a dense NumPy array
housing_cat_1hot = encoder.fit_transform(housing_cat, sparse_output=False)

'''----------------------------------Feature scaling--------------------------------'''
'''ML algorithms don't perform well when the input numerical attributes have very different scales.'''
'''
MinMaxScaler(), affected by mistaken max values, range in 0-1 or diy else
StandardScaler(), not affected by wrong max values, but not in range
'''

'''Custom transformers and pipeline them, both with numerical and non-numerical values.'''


