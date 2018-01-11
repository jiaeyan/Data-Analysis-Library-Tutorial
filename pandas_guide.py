import pandas as pd, numpy as np
from pandas import Series, DataFrame

'''-----------------------------01. Series----------------------------'''
'''A dict (with a name), with values corresponding to labels/indexes.'''

'''[Construction]'''
obj = pd.Series([4, 7, -5, 3]) # or use keyword args to indicate 'name', 'index' and 'values'
print(obj)
print(obj.values)
print(obj.index) # default range index, seen as labels of dict, index object in pandas is immutable
obj.name = 'ID' # assign "name" to series and its index
obj.index.name = 'names'
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan'] # alter indices/labels in place, but not work for values

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000} # map from index/label to value
obj3 = pd.Series(sdata) # pass a dict to series, key->index, forced to be sorted
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states) #DIY the order of keys
print()

'''[Modification]'''
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print(obj2['a']) # check value by label
print()
obj2['d'] = 6    # change value by label, in place
print(obj2[['c', 'a', 'd']])
print()

'''[Operation]'''
print(obj2[obj2 > 0])
print()
print(obj2 * 2) # this operation returns a new list, not in place
print()
print('b' in obj2) # only check if LABEL is in, not value
print(obj3 + obj4) # automatically aligns by index label in arithmetic operations
print()

'''[Reindex]'''
'''Pass a list of index, return a new object.'''
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0) # values will be fixed with existed index
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')

'''[Drop]'''
'''Dropping one or more entries from an axis.'''
obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c') # drop one item
obj.drop(['d', 'c']) # drop several items

'''[Indexing, Selection, and Filtering]'''
'''Retrieve a value'''
obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
print(obj['b'] == obj[1]) # check value either by index or integer index
obj[obj < 2]
obj[1:3] # the end point is exclusive if slicing with integer index
obj['b':'c'] # the end point is inclusive if slicing with labels
obj['b':'c'] = 5 # if interval length = 1, set one cell; if >1, set all cells as the give value

'''[Data Alignment]'''
'''Lengths can be different.'''
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1 + s2 # only operate on common indexed values, pad any uncommon ones with NaN (single side)

'''[Sorting and Ranking]'''
obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index() # sort lexicographically by index
obj.sort_values() # sort by values

'''[Unique Values, Value Counts, and Membership]'''
obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique()

'''Compute value frequency.'''
obj.value_counts()
pd.value_counts(obj.values, sort=False) # return a series of frequencies indexed by values,

'''data filtering'''
mask = obj.isin(['b', 'c']) # return a boolean series indicating if the value occurs
obj[mask] # return a series with values in mask, indexed by index in obj (not continuous)

'''compute a histogram on multiple related columns'''
data = pd.DataFrame({'Qu1': [1, 3, 4, 3, 4], 'Qu2': [2, 3, 1, 2, 3], 'Qu3': [1, 5, 2, 4, 4]})
result = data.apply(pd.value_counts).fillna(0) # the row labels in the result are the distinct values occurring in all of the col‐ umns. The values are the respective counts of these values in each column.

'''------------------------------02. DataFrame-------------------------------------'''
'''A collection of series as columns, each series can be different data type.'''

'''[Construction]'''
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data) # keys become column names and sorted, list items become values, index is assigned automatically and 
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop'], index=['one', 'two', 'three', 'four','five', 'six']) # force column name order

pop = {'Nevada': {2001: 2.4, 2002: 2.9}, # outer dict keys as the column names and the inner keys as the row indices
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = pd.DataFrame(pop)
frame3.values #check values in ndarray
frame3.index.name = 'year' # index has a name
frame3.columns.name = 'state' # column names also has a general name
print(frame3.columns)
print('print colums above')

'''[Modification]'''
print(frame2['state']) # retrieve a series/column by column name/key
print(frame2.state)
frame2['debt'] = 16.5 #create a new column if not included, value can be a scalar, array or a series (aligned, NAN if not matched)
print(frame2.loc['three']) # retrieve a row by index
del frame2['debt']

'''[Operation]'''
frame3.T # transpose the DataFrame (swap rows and columns)
print(frame3.index.unique())

'''[Reindex]'''
'''Work for both index and columns.'''
frame = pd.DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)
 
# frame.loc[['a', 'b', 'c', 'd'], states] # reindex index and columns in one line by 'loc'
 
'''[Drop]'''
'''Return new object'''
data = pd.DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], columns=['one', 'two', 'three', 'four'])
data.drop(['Colorado', 'Ohio']) # drop values from the row labels/index (axis 0)
data.drop('two', axis=1) # drop values from the columns by passing axis=1 or axis='columns'
data.drop(['two', 'four'], axis='columns')
obj.drop('c', inplace=True) # operate in place
  
'''[Indexing, Selection, and Filtering]'''
'''Retrieve a value'''
print(data['two']['Colorado']) # fix a column then fix a row, return the VALUE, rather than a pandas object

print(data.loc['Colorado', 'two']) # fix a row then fix a column, return the VALUE, rather than a pandas object
print(data.loc['Colorado', ['two']]) # return a pandas object of single one value
print(data.iloc[1, 1])

print(data.at['Colorado', 'two']) # select a value by ROW label then COLUMN label
print(data.iat[1, 1]) # select a value by ROW integer index then COLUMN integer index


'''Retrieve a column.'''
'''Values indexed by index'''
data = pd.DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], columns=['one', 'two', 'three', 'four'])
data['two'] # check a series/column by its name/key, integer index retrieval like data[1] doesn't work
data.two
data[:2] # get first 2 columns; integer index only works for slicing
data < 5 # return an object with boolean values for the condition
data[data < 5] = 0 # set all qualified values to 0

'''Retrieve a row.'''
'''Values indexed by column keys'''
# print(data.loc['Colorado', ['two', 'three']]) # select a row of 'Colorado', and retrieve some values in the row according to their column name/key
data.loc[:'Utah', 'two'] # select rows until 'Utah', and only column 'two'
data.iloc[[1, 2], [3, 0, 1]] # select several rows
data.iloc[2, [3, 0, 1]] # select by integer index with the same mechanism
data.iloc[:, :3][data.three > 5]

'''[Data Alignment]'''
'''Sizes can be different'''
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'), index=['Ohio', 'Texas', 'Colorado'])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1 + df2 # only operate on values that have the same index and column in both tables, otherwise show 'NaN'; may create new column/index
df1.add(df2, fill_value=0) # fill all nan values with 0.

'''DataFrame + Series'''
s1 = df1.iloc[0] # indexed with column keys
df1 + s1 # row broadcasting according to column keys

s2 = df1['b'] # indexed with index
df1.add(s2, axis='index') # column broadcasting along index

'''[Function Application and Mapping]'''
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])

'''column/row wise function'''
f = lambda x: x.max() - x.min() # DIY a function
frame.apply(f) # apply on columns, and return a series indexed by column keys
frame.apply(f, axis='columns') # apply on rows, and return a series indexed by index

frame.sum() # sum values along columns, indexed by column keys, exclude nan
frame.sum(axis = 1) # sum values along rows, indexed by index, exclude nan


'''element wise function'''
np.abs(frame) # all numpy functions work for the dataframe object

format = lambda x: '%.2f' % x
frame.applymap(format) # applymap for dataframe to do element wise function
frame['e'].map(format) # map() for series to do element wise function

'''[Sorting and Ranking]'''
frame = pd.DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'], columns=['d', 'a', 'b', 'c'])
frame.sort_index() # sort by index
frame.sort_index(axis=1, ascending=False) # sort by column keys

frame.sort_values(by='b') # sort by values of certain columns
frame.sort_values(by=['a', 'b'])
# '''------------------------------03. Reindexing-------------------------------------'''
#  
#  


