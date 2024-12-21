# Pasupuleti-Charan-Kumar
# Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("/content/Social_Network_Ads.csv")
dataset.head()
dataset.isna().sum()
x = dataset.iloc[:, [2, 4]].values
y = dataset.iloc[:, 4].values
x
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
x_train.shape
x_test.shape
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train
x_test
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
