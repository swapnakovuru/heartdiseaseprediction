import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv("heart.csv")
print(data.head())
dataset = pd.read_csv('heart.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(random_state=0)
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
print(X_test)
print(y_pred)
from sklearn.svm import SVC
model1 = SVC(kernel='linear')(svc=svm.SVC(kernel='rbf'))
model1.fit(iX_train,y_train)
svm_predictions =model1.predict(features_test)


         
