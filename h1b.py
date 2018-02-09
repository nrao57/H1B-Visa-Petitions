#Predict whether a person will be certified or denied
import pandas as pd
import matplotlib.pyplot as plt
import h1bfunctions
from h1b_tensor import DeepNN
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime

start_time = datetime.now() #for timing the script

#dataset path
data_set_path = 'h1b_kaggle.csv'

#extract the data 
chunks = 10000 #number of data rows want to use
data_set = h1bfunctions.get_data(data_set_path, chunk = chunks)

#clean the data
new_data = h1bfunctions.format_clean(data_set)

#split the data
xs = new_data.drop('CASE_STATUS', axis = 1)  
ys = new_data['CASE_STATUS']
X_train, X_test, y_train, y_test = train_test_split(xs,ys, test_size=0.30)

#summary statistics
print(xs.describe())
print(ys.describe())

#Transform and Scale Data
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)

#Logistic Regression
LR = LogisticRegression()
LR.fit(scaler.transform(X_train), y_train)
y_pred=LR.predict(scaler.transform(X_test))

#Deep Neural Net
y_train_codes = y_train.astype('category').cat.codes #convert y_training labels (certified = 0, denied = 1)
y_test_codes = y_test.astype('category').cat.codes
dnn_acc = DeepNN(X_train.values, y_train_codes.values.astype(int), X_test.values, y_test_codes.values.astype(int))

#Accuracy Measurements
print("\n"+"The accuracy of the Logistic Model is {}".format(accuracy_score(y_test,y_pred)))
print("The accuracy of the Deep Neural Net for training data is {}".format(dnn_acc[0]))
print("The accuracy of the Deep Neural Net for test set is {}".format(dnn_acc[1]))

print("\n"+"Runtime: {}".format(datetime.now()-start_time)) 

#Visuallizations
#h1bfunctions.visualize(xs)
