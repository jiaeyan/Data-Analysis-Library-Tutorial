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

05. Select a model and train
    a. cross validation training

06. Fine Tune the Parameters of the Model
    a. grid search with different combinations of parameters
    b. check the importance of each feature, drop the useless ones

07. Test

08. Save the model
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

'''----------------------------------Train and Evaluate-------------------------------'''
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
# train a model with prepared and cleaned data and relative labels
lin_reg.fit(housing, ['put label list here'])

# take a little training set to play with
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
# clean and prepare the small train set
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:\t", lin_reg.predict(some_data_prepared))
#Predictions: [ 303104. 44800. 308928. 294208. 368704.]
print("Labels:\t\t", list(some_labels))

# measure this regression model’s RMSE on the whole training set
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions) 
lin_rmse = np.sqrt(lin_mse)

'''Evaluate with Cross-Validation'''
from sklearn.model_selection import cross_val_score
# cv: how many subsets you'd like to apply
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# return an array of rmse result from each cross validation training
rmse_scores = np.sqrt(-scores)

# there are 10 results in 'scores', thus calculate their mean and std
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(rmse_scores)


'''--------------------------------Fine tune the model-----------------------------'''
'''Grid Search'''
# evaluate all the possible combinations of hyperparameter values, using cross-validation.
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# a list of dicts of combinations of parameters you'd like to test
# one dict is one combination, here is two, and there are 3*4 + 2*3 = 18 combinations
# the key in each dict should match the parameter names of given model
param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

# define a model
forest_reg = RandomForestRegressor()

# define an instance of the estimator, feed in the parameters
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error')

# train the model
grid_search.fit(housing_prepared, housing_labels)

# check the best parameters
grid_search.best_params_

# get the best estimator directly
grid_search.best_estimator_

# get the evaluation scores on the hold-out set of cross validation
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

'''Check the importance of each feature'''
# only return a list of figures, no correspoding feature names
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)



'''--------------------------------------Test------------------------------------------'''
# get the best model with best parameters
final_model = grid_search.best_estimator_

# get the test data, separating the label column
# test data
X_test = strat_test_set.drop("median_house_value", axis=1)
# test data's labels
y_test = strat_test_set["median_house_value"].copy()

# clean the test data
X_test_prepared = full_pipeline.transform(X_test)

# make predictions on test data and returns a list of results
final_predictions = final_model.predict(X_test_prepared)

# get the results
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 48,209.6


'''---------------------------------Save the model----------------------------------'''
# a model example
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

'''Store with Python's pickle.'''
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)

'''Store with Sklearn's joblib.'''
from sklearn.externals import joblib
joblib.dump(clf, 'filename.pkl') 
clf = joblib.load('filename.pkl')