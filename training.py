## Ali Akbar Naqvi
## Internship Project 1
## California House Price Prediction

import pandas as pd
import pickle

data = pd.read_csv("housing.csv")

data.dropna(inplace=True)


from sklearn.model_selection import train_test_split
x=data.drop(['median_house_value'], axis=1)
y=data['median_house_value']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2) ## 20% Data reserved for testing

train_data= x_train.join(y_train)


train_data=train_data.join(pd.get_dummies(train_data.ocean_proximity, dtype=int)).drop(['ocean_proximity'], axis=1)

test_data=x_test.join(y_test)

test_data=test_data.join(pd.get_dummies(test_data.ocean_proximity, dtype=int)).drop(['ocean_proximity'], axis=1)

from sklearn.linear_model import LinearRegression

x_train,y_train=train_data.drop(['median_house_value'],axis=1),train_data['median_house_value']
reg = LinearRegression()
reg.fit(x_train,y_train)

with open('CHPPM.pkl', 'wb') as file:
    pickle.dump(reg,file)

x_test,y_test=test_data.drop(['median_house_value'],axis=1),test_data['median_house_value']

print(reg.score(x_test,y_test))

