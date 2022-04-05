import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
titanic_data = pd.read_csv('train.csv')
print (titanic_data.head())
print (titanic_data.isnull().sum())
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
print(titanic_data['Embarked'].mode())
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
#print (titanic_data.isnull().sum())
titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
print (titanic_data.head())


X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived']) #target
Y = titanic_data['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

#Checking for a Random Person
input_data = (3,0,35,0,0,8.05,0)
input_data_as_numpy_array = np.asarray(input_data) # change the input_data to numpy
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
if prediction[0]==0:
    print("Dead")
if prediction[0]==1:
    print("Alive")