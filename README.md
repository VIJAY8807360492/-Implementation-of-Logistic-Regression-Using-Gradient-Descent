# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: vijay k
RegisterNumber: 24901153

import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("/content/Placement_Data (1).csv")
data1 = data.copy()

# Drop unnecessary columns
data1 = data.drop(['sl_no', 'salary'], axis=1)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Define features (X) and target variable (Y)
X = data1.iloc[:, :-1].values
Y = data1["status"].values

# Initialize weights randomly
theta = np.random.randn(X.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Gradient Descent function
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

# Train the model using Gradient Descent
theta = gradient_descent(theta, X, Y, alpha=0.01, num_iterations=1000)

# Prediction function
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

# Make predictions on training data
y_pred = predict(theta, X)

# Calculate accuracy
accuracy = np.mean(y_pred.flatten() == Y)
print("Accuracy:", accuracy)

# Print predictions and actual values
print("Predicted:\n", y_pred)
print("Actual:\n", Y)

# Predict for a new input
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_pred_new = predict(theta, xnew)
print("Predicted Result:", y_pred_new)

 
*/
```

## Output:

![image](https://github.com/user-attachments/assets/0ba86887-a7f9-4dcd-bda2-4885ed202b5b)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

