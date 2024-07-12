import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
p = pd.read_csv("C:\\Users\\DELL\\Desktop\\python projects and achievements\\advertising.csv")
print( p.TV.corr(p.Sales))
print( p.Radio.corr(p.Sales))
print( p.Newspaper.corr(p.Sales))
m1=p.TV.mean()
sd1=p.TV.std()
low=m1-3*sd1
high=m1+3*sd1
m3=p.Newspaper.mean()
sd3=p.Newspaper.std()
low=m3-3*sd3
high=m3+3*sd3
p[(p.Newspaper<low)|(p.Newspaper>high)]
p=p[(p.Newspaper>low)&(p.Newspaper<high)]
model = LinearRegression()
X = p.drop('Sales', axis='columns')
y = p['Sales']
model.fit(X,y)
model.score(X,y)
print(model.coef_)
print(model.intercept_)
print(model.predict([[44,21,19]]))
