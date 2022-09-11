# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:51:13 2022

@author: SharwinA
"""
#importing Packages

import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns


#Dataset Loading
data = "http://bit.ly/w-data"
data = pd.read_csv(data)
print(data.head(10))

#Relationship b/w variables
sns.scatterplot(x = 'Hours', y = 'Scores', data= data)
plt.title('Hours vs Percentage')

#preparing test and training dataset
x = data.loc[:, ['Hours']].values
y = data.loc[:,['Scores']].values

print('Hours',x)  
print('Scores',y)

#Splitting the data into test and training
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Training the model
from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train, y_train)

#Evaluating the model by plotting regression line
#y = mx+c
line = model.coef_*x+model.intercept_
plt.scatter(x,y)   
plt.plot(x, line)  
plt.show()

#Predicting the results
y_pred = model.predict(X_test)
print(y_pred)


#We want to predict scores for 9.25 hours Therefore, 
pred = model.predict([[9.2]])
print("No of Hours = {}".format(9.2))
print("Predicted Score = {}".format(pred[0]))

#Checking the accuracy of the model by MAE and R2 score
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

from sklearn.metrics import r2_score
r1 = r2_score(y_test, y_pred)
print("R2 score : {:.2f}%".format(r1*100))
