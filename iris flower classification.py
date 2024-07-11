import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
p = pd.read_csv("C:\\Users\\DELL\\Desktop\\python projects and achievements\\IRIS.csv")
X = p.iloc[:, :-1]
y = p.iloc[:, -1]
le = LabelEncoder()
y_encod = le.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encod, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(model.predict([[5,3,1.4,0.6]]))
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
model.predict([[5,3,1.4,0.6]])


