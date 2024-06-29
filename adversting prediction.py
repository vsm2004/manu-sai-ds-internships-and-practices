import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
p = pd.read_csv("C:\\Users\\DELL\\Desktop\\python projects and achievements\\advertising.csv")
model = LinearRegression()
X = p.drop('Sales', axis='columns')
y = p['Sales']
model.fit(X,y)
model.score(X,y)
model.predict([[44,21,19]])
print(model.coef_)
print(model.intercept_)
