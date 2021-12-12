import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

"""Data collection and Processing"""

gold_data = pd.read_csv('/content/gld_price_data.csv')

#print first five rows of the data frame
gold_data.head()

# print last 5 rows of the DataFrame
gold_data.tail()

#No. of rows and column
gold_data.shape

#Getting some basic information about the data
gold_data.info()

#Checking the number of missing values
gold_data.isnull().sum()

#Getting statistical measures of the data
gold_data.describe()

"""Correlation: (Tells us which column is related to which column)
1. Positive Correlation
2. Negative Correlation
"""

correlation = gold_data.corr()

#Constructing a heat map to understand the correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Greens')

#correlation values of GLD
print(correlation['GLD'])

#Check the distribution of the gold price
sns.distplot(gold_data['GLD'])

"""splitting the feature and the target"""

X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']

print(X)

print(Y)

"""Splitting into training data and test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=2)

"""Model training :
Random Forest Regressor. #It is an ensemble model(Uses many models). We use multiple decision trees.
"""

regressor = RandomForestRegressor(n_estimators=100)

#training the model
regressor.fit(X_train, Y_train)



"""Model Evaluation"""

#prediction on test data
test_data_prediction = regressor.predict(X_test)

print(test_data_prediction)

"""Now we have to compare the predicted values with the actual values."""

#R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error :", error_score)

"""Compare the actual values and predicted values in a plot"""

Y_test = list(Y_test)

plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green',label='Predicted Value')
plt.title('Actual Price Vs Predictedd Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show

