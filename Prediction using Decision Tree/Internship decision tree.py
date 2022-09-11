# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 12:13:58 2022

@author: SharwinA
"""

import os
import pandas as pd
import matplotlib.pyplot as plt  

#Dataset Loading
os.chdir('S:\LAB FILES\Internship')
data = pd.read_csv('Iris(1).csv')
print(data.head(10))    
print(data.describe())

data = data.drop(columns = 'Id')

#x = data.iloc[:, [0,1,2,3]].values
#y = data.iloc[:, 4].values
x = data.drop("Species", axis=1)
y= data["Species"]

print(x)
print(y)
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
model= DecisionTreeClassifier()
model.fit(x,y)

y_pred = model.predict(X_test)
print(y_pred)

from sklearn import tree
#feature = data.drop(columns = 'Species')

classname = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
# Visualize the graph
fig = plt.figure(figsize=(25,20))
fig_ = tree.plot_tree(model, feature_names=x.columns, 
                      class_names=classname, 
                      filled=True)

#Checking the accuracy of the model by accuracy score on X_test data

from sklearn.metrics import accuracy_score
a1 = accuracy_score(y_test, y_pred)
print("Accuracy score : {:.2f}%".format(a1*100))

#Testing on random data points except from dataset
Testing = [[6.4,3.0,4.5,1.5],
           [6.5,2.8,4.6,1.3],
           [5.0,2.5,-0.5,1.4],
           [6.0,2.2,5.1,1.1],
           [6.0,3.0,1.5,-2.8]]

print(model.predict(Testing))
