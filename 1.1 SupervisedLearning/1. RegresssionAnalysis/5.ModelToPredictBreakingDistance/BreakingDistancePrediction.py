import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('data.csv')

# print('Print Data \n', df)
 
# df.info()

# corr = df['Speed'].corr(df['BrakingDistance'])

# print('Correlation: ', corr)

plt.plot(df['Speed'], df['BrakingDistance'])
plt.xlabel('Speed')
plt.ylabel('BrakingDistance')
plt.title('Speed vs BrakingDistance')
plt.show()

speed = df.drop(columns = ['BrakingDistance'], axis=1)
brakingDistance = df['BrakingDistance']

# print(brakingDistance)

# for degree in range(1,10): 
#   poly = PolynomialFeatures(degree=degree)
#   speed_square = poly.fit_transform(speed)

#   # print(speed_square)

#   model = LinearRegression()
#   model.fit(speed_square, brakingDistance)

#   output = model.predict(poly.fit_transform([[115]]))
#   print(f'Degree {degree} Output is {output}')