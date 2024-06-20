# Importing Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv(r"C:\Users\abang\OneDrive\Desktop\Data_Hub\datasets\Copy of sonar data.csv",header=None)
sonar_data

sonar_data.shape

sonar_data.describe()
sonar_data[60].value_counts()
sonar_data.groupby(60).mean()

x= sonar_data.drop(columns=60,axis=1)
y = sonar_data[60]

print(x)
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)

print(x.shape,x_train.shape,x_test.shape)

model = LogisticRegression()
model.fit(x_train,y_train)

train_data_pred = model.predict(x_train)
train_data_pred_accuracy = accuracy_score(train_data_pred,y_train)

print(train_data_pred_accuracy*100)


# evaluation on test data
test_pred = model.predict(x_test)
test_pred_accuracy = accuracy_score(test_pred,y_test)

print(f"accuracy of test data: {test_pred_accuracy*100}")


input_data = (0.0249,0.0119,0.0277,0.0760,0.1218,0.1538,0.1192,0.1229,0.2119,
              0.2531,0.2855,0.2961,0.3341,0.4287,0.5205,0.6087,0.7236,0.7577,
              0.7726,0.8098,0.8995,0.9247,0.9365,0.9853,0.9776,1.0000,0.9896,
              0.9076,0.7306,0.5758,0.4469,0.3719,0.2079,0.0955,0.0488,0.1406,
              0.2554,0.2054,0.1614,0.2232,0.1773,0.2293,0.2521,0.1464,0.0673,
              0.0965,0.1492,0.1128,0.0463,0.0193,0.0140,0.0027,0.0068,0.0150,
              0.0012,0.0133,0.0048,0.0244,0.0077,0.0074)
    
input_data_as_nparray = np.asarray(input_data)
data_reshaped = input_data_as_nparray.reshape(1,-1)

prediction = model.predict(data_reshaped)
print(prediction)

import pickle

with open('rock vs mine.pkl','wb') as file:
    pickle.dump(model,file)
    
with open('rock vs mine.pkl','rb') as file:
    loaded_model = pickle.load(file)
    
loaded_model.predict(x_test)


input_data = (0.0249,0.0119,0.0277,0.0760,0.1218,0.1538,0.1192,0.1229,0.2119,
              0.2531,0.2855,0.2961,0.3341,0.4287,0.5205,0.6087,0.7236,0.7577,
              0.7726,0.8098,0.8995,0.9247,0.9365,0.9853,0.9776,1.0000,0.9896,
              0.9076,0.7306,0.5758,0.4469,0.3719,0.2079,0.0955,0.0488,0.1406,
              0.2554,0.2054,0.1614,0.2232,0.1773,0.2293,0.2521,0.1464,0.0673,
              0.0965,0.1492,0.1128,0.0463,0.0193,0.0140,0.0027,0.0068,0.0150,
              0.0012,0.0133,0.0048,0.0244,0.0077,0.0074)
    
input_data_as_nparray = np.asarray(input_data)
data_reshaped = input_data_as_nparray.reshape(1,-1)

prediction = loaded_model.predict(data_reshaped)
print(prediction)