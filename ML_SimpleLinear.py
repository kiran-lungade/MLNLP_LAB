import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("houseprices.csv")
#print(data)

x=data.iloc[0:100,1:2].values
#print(x)

y=data.iloc[0:100,:1].values
#print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
#print(x_train)
#print(len(x_train))

from sklearn.linear_model import LinearRegression
reg=LinearRegression()

reg.fit(x_train,y_train)

predicted=reg.predict(x_test)

from sklearn.metrics import mean_squared_error
print("mean squared error is: \n",mean_squared_error(y_test,predicted))
print("predicted data by model \n",predicted)
print("Actual data \n",y_test)

r2score=reg.score(x_test,y_test)
print(r2score)

print("weight or slope M is : \n",reg.coef_)

print("intercept C is :\n",reg.intercept_)

plt.scatter(x_test,y_test)
plt.plot(x_test, predicted, color = "g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
