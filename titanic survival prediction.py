import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import confusion_matrix
p = pd.read_csv("C:\\Users\\DELL\\Desktop\\python projects and achievements\\Titanic-Dataset.csv")
q = p[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
q['Age'].fillna(q['Age'].mean(), inplace=True)
dummies = pd.get_dummies(q[['Sex', 'Embarked']])
merged = pd.concat([q.drop(['Sex', 'Embarked'], axis=1), dummies], axis=1)
X = merged.drop('Survived', axis='columns')
y = merged['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=5)
model = LogisticRegression(C=0.01, max_iter=15000)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(accuracy)
y_pred=model.predict(X_test)
print(y_pred)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,3))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')


