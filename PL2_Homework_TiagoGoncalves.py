# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:21:38 2017

@author: tiago
"""

#Homework PL2 Tiago Gonçalves
#6. Homework
#Imported libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#6.1 - Graded Homework: Training and Testing data.

#print_predictions_and_labels function (Provided by Professor)
def print_predictions_and_labels(array_preds_labels):
    for predicted_pair in array_preds_labels:
        prediction = predicted_pair[0]
        label = predicted_pair[1]
        print('Prediction', prediction, 'Label', label)

#Part 1
#Read Data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
                 header=None)
df.columns = ['#pregnant', 'glucose', 'blood_pressure', 'thickness', 'insulin', 'body_mass_idx', 
              'pedigree', 'age', 'label']
df.head()

#Separate numpy arrays for labels and features
dataset = np.array(df)
print(df.shape)

features = dataset[:, :8]
labels = dataset[:, -1]

#Split data into train set and test set (75% for test set!)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42 )

#Fit model with sklearn (Training classifier)
#Use test set
clf = GaussianNB()
clf.fit(X_train, y_train)
test = clf.predict(X_test)

#Store test predictions and labels in a numpy array
new_array = np.concatenate([test.reshape(-1, 1), y_test.reshape(-1, 1)], axis=1)
print_predictions_and_labels(new_array)

#Part 2
#Alternative 1: Dividing the number of correct predictions by the number of total predictions.
accuracy = accuracy_score(test, y_test, normalize=False)/len(y_test)
print('Accuracy A1: \n', accuracy)
#Alternative 2: Using accuracy_score()
accuracy_sklearn = accuracy_score(test, y_test)
print('Accuracy A2: \n', accuracy_sklearn)
# Alternative 3: Using .score method
accuracy_sklearn_2 = (clf.fit(X_train, y_train)).score(X_test, y_test)
print('Accuracy A3: \n', accuracy_sklearn_2)

##############################################################################
#6.2 Graded Homework: Simple Linear Regression
#First Part
#Load diabetes dataset and fit the entire data as given in the example
diabetes_dataset = load_diabetes(return_X_y=True)

model = LinearRegression()
model.fit(diabetes_dataset[0], diabetes_dataset[1])

#Fit the regressor with 70% of the dataset
diX_train, diX_test, diy_train, diy_test = train_test_split(diabetes_dataset[0], diabetes_dataset[1], test_size=0.30, random_state=42)

#Train the model with this new dataset
nmodel = LinearRegression()
nmodel.fit(diX_train, diy_train)

#Second Part
#Compute predictions of the model on the test data
diypred = nmodel.predict(diX_test)

#Get the Mean Squared Error
mserror = mean_squared_error(diy_test, diypred)
print('Mean Squared Error is: \n', mserror)

#Get variance score (extra)
cov = r2_score(diy_test, diypred)
print('Variance is: \n', cov)

# The coefficients (extra)
print('Coefficients: \n', nmodel.coef_)

##############################################################################
#6.3 Ungraded Homework - k neighboors
from sklearn import datasets
from collections import Counter

def distance(x,y):
    return np.linalg.norm(x-y)

#Implement from scratch
#Creating the train block of KNN Classifier

def knntrain(Xknn_train, yknn_train):
    return

#Creating the predict block of KNN Classifier

def predictknn(Xknn_train, yknn_train, Xknn_test, k):
    knndistances = []
    knntargets = []
    
    for i in range(len(Xknn_train)):
        #Compute Euclidean Distance
        d = np.sqrt(np.sum(np.square(Xknn_test - Xknn_train[i, :])))
        #Add it to list of distances
        knndistances.append([d, i])
    
    #Sort the list
    knndistances = sorted(knndistances)
    
    #Make a list to have k neighbor's targets
    for i in range (k):
        kindex = knndistances[i][1]
        knntargets.append(yknn_train[kindex])
        
    #Get and return most common knntarget
    return Counter(knntargets).most_common(1)[0][0]

# Define the KNN Fucntion that will loop through everything and perform the test using KNN Classifier

def KNN(Xknn_train, yknn_train, xknn_test, knnpreds, k):
    #Train on the input data
    knntrain(Xknn_train, yknn_train)
    
    #Then, we loop through all the observations
    for i in range(len(xknn_test)):
        knnpreds.append(predictknn(Xknn_train, yknn_train, xknn_test[i::], k))


#Testing the KNN Algorithm
#With a simple array
iris = datasets.load_iris()

subset = np.logical_or(iris.target == 0, iris.target == 1)

X = iris.data[subset]
y = iris.target[subset]

xnew = np.array([3.5, 2.5, 2.5, 0.75])

#Save obtained predictions
knnpreds = []

#KNN with k=3
KNN(X, y, xnew, knnpreds, 3)

#Transform the list into an array
knnpreds = np.asarray(knnpreds)

#Print predictions
print("Predicted values are: \n", knnpreds)

#With a test split
Xknn_train, Xknn_test, yknn_train, yknn_test = train_test_split(X, y, test_size=0.30, random_state=42)

kpreds = []
KNN(X, y, xnew, kpreds, 3)

kpreds = np.array(kpreds)

#Print predictions
print("Predicted values are: \n", kpreds)
    
#With sklearn library
# loading library
from sklearn.neighbors import KNeighborsClassifier

#Test Split
Xknn_train, Xknn_test, yknn_train, yknn_test = train_test_split(X, y, test_size=0.30, random_state=42)

#KNN with k=3
knn = KNeighborsClassifier(n_neighbors=3)

#Fitting the model
knn.fit(Xknn_train, yknn_train)

#Predict the response
pred = knn.predict(Xknn_test)
print("Predicted values are: \n", pred)

#Evaluate accuracy
print("Accuracy for KNN is: \n", accuracy_score(pred, yknn_test))