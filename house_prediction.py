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
#as the data is skewed in the histograms, we will tske the logarithms of the data
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
x_train, y_train = train_data.drop('median_house_value', axis=1), train_data['median_house_value']
reg = LinearRegression()
reg.fit(x_train,y_train)
# %%
train_data
# %%
