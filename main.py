import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


dataframe = pd.read_csv('Medical_insurance.csv')

#print(dataframe.head())

# Data Preprocessing
print(dataframe.isnull().sum()) #No missing values

# Data analysis
print(dataframe.describe()) # Statistical summary

# Visualizing the data
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.displot(dataframe['age'], kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
#plt.show()

#Gender Coloumn
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', data=dataframe)
plt.title('Sex Distribution')
#plt.show()

# BMI Column
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.displot(dataframe['bmi'], kde=True, color='blue')
plt.title('BMI Distribution')
plt.xlabel('bmi')
plt.show()

#similarly for other columns

#Encoding the cateogorical variables

dataframe.replace({'sex' : { 'male': 0 , 'female': 1 }}, inplace=True)
dataframe.replace({'smoker' : { 'yes': 1 , 'no': 0 }}, inplace=True)
dataframe.replace({'region' : { 'southwest': 0 , 'southeast': 1, 'northwest': 2, 'northeast': 3 }}, inplace=True)


# Splitting the dataset into features and target variable

X = dataframe.drop(columns='charges', axis=1)
y = dataframe['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model training
model = LinearRegression()
model.fit(X_train, y_train)

#model prediction on training data
y_train_pred = model.predict(X_train)
r2_train = metrics.r2_score(y_train, y_train_pred)
# Model prediction on testing data
y_pred = model.predict(X_test)
r2_test = metrics.r2_score(y_test, y_pred)
print(f"R^2 score on training data: {r2_train}")
print(f"R^2 score on testing data: {r2_test}")

#Building a predictive system

input_data = (19,0,24.6,1,0,0)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1) # can even use reshape(1, 6) 

prediction = model.predict(input_data_reshaped)
print( "The prediction for the given input is", prediction)