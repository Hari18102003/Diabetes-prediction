# Importing the Dependencies 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data processing 
diabetes_data = pd.read_csv("diabetes.csv")

#first 5 data
diabetes_data.head()
diabetes_data.shape

#statistical data
diabetes_data.describe()

# separate labels and features 
X = diabetes_data.drop(columns="Outcome",axis=1)
Y = diabetes_data["Outcome"]
print(X)
print(Y)

# Standardize the data 
scaler = StandardScaler()
std_data = scaler.fit_transform(X)
print(std_data)

# Train and Test data splitting 
X = std_data
Y = diabetes_data["Outcome"]

print(X)
print(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape)
print(X_train.shape)
print(X_test.shape)

# Model Training 
model = LogisticRegression()
model.fit(X_train,Y_train)

# Testing the accuracy score with training data 
X_train_prediction = model.predict(X_train)
score = accuracy_score(X_train_prediction,Y_train)
print(score)

X_test_prediction = model.predict(X_test)
score = accuracy_score(X_test_prediction,Y_test)
print(score)

# building prediction system 
input_data = (1,85,66,29,0,26.6,0.351,31)
input_data_as_array = np.asarray(input_data)
input_data_reshaped = input_data_as_array.reshape(1,-1)
scaled_input = scaler.transform(input_data_reshaped)

prediction = model.predict(scaled_input)
print(prediction)

if prediction[0] == 0:
  print("Non Diabetic person")
else:
  print("Diabetic person")