import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('data.csv')
x = df.drop(columns=['b'], axis=1)
y = df['b']

print(x)

plt.plot(df['a'], df['b'])
plt.xlabel('a')
plt.ylabel('b')
plt.title('a vs b')
plt.show()

poly = PolynomialFeatures(degree=2)
x_sqaure = poly.fit_transform(x)

print(x_sqaure)
model = LinearRegression()
model.fit(x_sqaure, y)

output = model.predict(poly.fit_transform([[634]]))
print(f'For x=634, Predicted value of y = {output}')