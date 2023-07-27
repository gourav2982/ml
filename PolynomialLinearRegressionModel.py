import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Dataset/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

lin_regre1 = LinearRegression()
lin_regre1.fit(X, y)

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_regre2 = LinearRegression()
lin_regre2.fit(X_poly, y)

plt.scatter(X, y, color='red')
plt.plot(X, lin_regre1.predict(X), color='blue')
plt.title('Level vs Salary (Training set)')
plt.ylabel('Years of Experience')
plt.xlabel('Salary')
plt.show()


plt.scatter(X, y, color='red')
plt.plot(X, lin_regre2.predict(X_poly), color='blue')
plt.title('Level vs Salary (Training set)')
plt.ylabel('Years of Experience')
plt.xlabel('Salary')
plt.show()
