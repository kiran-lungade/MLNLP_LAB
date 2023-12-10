import pandas as pd 
import numpy as np 

data=pd.read_csv(r"Iris.csv")
#print(data)

x=data.iloc[:,:-1].values
#print(x)

y=data.iloc[:,-1].values
#print(y)

#convert target variable values into numeric form
from sklearn.preprocessing import LabelEncoder
y=LabelEncoder().fit_transform(y)  
#print(y) 

#split train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
#print(x_train.shape)
#print(x_test.shape)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print("actual values of dataset",y_test)
print("predicated values given by model",y_predict)

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
accuracy=accuracy_score(y_test,y_predict)*100
print("accurayc of Bayes Classifier is ",accuracy)

cf=confusion_matrix(y_predict,y_test)
print(cf)

f1=f1_score(y_predict,y_test,average='weighted')
print("f1 score is ",f1)