#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#this is a file
#%%
df = pd.read_csv('/Users/swarnim/Desktop/housing.csv') #loading the file into our notebook
# %%
df
# %%
df.info() #checking for null values
# %%
df.dropna(inplace=True) #to drop the null values and () saves into the original file
# %%
from sklearn.model_selection import train_test_split #to split the data into train and test

x = df.drop('median_house_value', axis=1) #actual dataframe without target var -- median etc
y = df['median_house_value'] 
# %%
x
# %%
y
# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #splitting the data
# %%
train_data = x_train.join(y_train) #joining the x_train and y_train
# %%
train_data
# %%
train_data.hist(figsize=(15,8)) #histogram
# %%
#as the data is skewed in the histograms, we will take the logarithms of the data
train_data['total_rooms'] = np.log(train_data['total_rooms']+1) #+1 to avoid 0
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms']+1)
train_data['population'] = np.log(train_data['population']+1)
train_data['households'] = np.log(train_data['households']+1)
# %%
train_data.hist(figsize=(15,8))
# %%
train_data.ocean_proximity.value_counts() #to get the value count of the strings in ocean_proximity column
# %%
#now we will change the numbers to 1 or 0 meaning yes or no
pd.get_dummies(train_data.ocean_proximity)
# %%
#now we will join the last data to the og data
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop('ocean_proximity', axis=1)
# %%
train_data
# %%
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm')
# %%
plt.figure(figsize=(15,8))
sns.scatterplot(x='longitude', y='latitude', hue='median_house_value', palette="coolwarm", data=train_data) 
#hue is median house value as now in the plot we can see different colour ranges of median house value
# %%
#feature engineering
train_data['bedroom_ratio'] = train_data['total_bedrooms']/train_data['total_rooms'] #bedroom ratio
train_data['household_ratio'] = train_data['households']/train_data['total_rooms'] #household ratio
# %%
from sklearn.linear_model import LinearRegression
x_train, y_train = train_data.drop('median_house_value', axis=1), train_data['median_house_value'] #dropping the target
reg = LinearRegression()
reg.fit(x_train,y_train)
# %%
train_data
# %%
#scaling the data
test_data = x_test.join(y_test)

test_data['total_rooms'] = np.log(x_test['total_rooms']+1)
test_data['total_bedrooms'] = np.log(x_test['total_bedrooms']+1)
test_data['population'] = np.log(x_test['population']+1)
test_data['households'] = np.log(x_test['households']+1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop('ocean_proximity', axis=1)

# %%
# Adding 'bedroom_ratio' and 'household_ratio' features to the test data
test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_ratio'] = test_data['households'] / test_data['total_rooms']

# %%
x_test, y_test = test_data.drop('median_house_value', axis=1), test_data['median_house_value']
# %%
x_test.drop(['bedroom','household'], axis=1, inplace=True)
# %%
x_test
# %%
reg.score(x_test, y_test)
# %%
#randon forest
from sklearn.ensemble import RandomForestRegressor #ensemble is a technique where we combine multiple models to get better results
forest = RandomForestRegressor()
forest.fit(x_train, y_train)
# %%
forest.score(x_test, y_test)
# %%
from sklearn.model_selection import GridSearchCV #to tune the hyperparameters
param_grid = {
    'n_estimators': [100,200,300],
    'min_samples_split': [2, 4, 6,8],
    'max_depth': [None, 4, 8]
  
    # you are tuning the 'n_estimators' (number of trees in a random forest) and 'max_features' (the number of features to consider for splitting at each node).
}

grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(x_train, y_train)
# %%
forest_dekhte_hai = grid_search.best_estimator_
# %%
forest_dekhte_hai.score(x_test, y_test)
# %%
from sklearn.ensemble import GradientBoostingRegressor
# %%
gb = GradientBoostingRegressor()
gb.fit(x_train, y_train)
# %%
gb.score(x_test, y_test)
# %%
from sklearn.model_selection import GridSearchCV #to tune the hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
  
    # you are tuning the 'n_estimators' (number of trees in a random forest) and 'max_features' (the number of features to consider for splitting at each node).
}
grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(x_train, y_train)
# %%
dk = grid_search.best_estimator_
# %%
dk.score(x_test, y_test)
# %%
from sklearn.svm import SVR
# %%
sv = SVR(kernel='linear', C=400)
# %%
sv.fit(x_train, y_train)
# %%
svm_predictions = sv.predict(x_test)
# %%
sv.score(x_test, y_test)
# %%
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# %%
mse = mean_squared_error(y_test, svm_predictions)
r2 = r2_score(y_test, svm_predictions)
mae = mean_absolute_error(y_test, svm_predictions)
# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1.0, 10.0]
}

svm_grid_search = GridSearchCV(sv, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
svm_grid_search.fit(x_train, y_train)

best_svm_model = svm_grid_search.best_estimator_
best_svm_predictions = best_svm_model.predict(x_test)
# %%
best_svm_predictions
# %%
best_svm_model.score(x_test, y_test)
# %%
