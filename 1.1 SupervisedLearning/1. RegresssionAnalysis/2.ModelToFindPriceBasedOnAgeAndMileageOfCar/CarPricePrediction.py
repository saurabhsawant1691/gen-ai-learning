import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('car_price_data.csv')

print('Column names in the DataFrame:', df.columns)
df.info()

corr_age_price = df['Age'].corr(df['Price'])
corr_mileage_price = df['Mileage'].corr(df['Price'])

print(f'Correlation between Age and Price:{corr_age_price}')
print(f'Correlation between Mileage and Price:{corr_mileage_price}')

# plt.scatter(df['Age'], df['Price'])
# plt.xlabel('Age of Car')
# plt.ylabel('Price of Car')
# plt.title('Age vs Price')
# plt.show()

# plt.scatter(df['Mileage'], df['Price'])
# plt.xlabel('Mileage of Car')
# plt.ylabel('Price of Car')
# plt.title('Mileage vs Price')
# plt.show()

model = LinearRegression()

X = df.drop('Price', axis=1)
y = df['Price']

model.fit(X, y)

input_data = pd.DataFrame([[5, 50000]], columns=['Age', 'Mileage'])

print(input_data)

predicted_price = model.predict(input_data)

print(f'Predicted price for a car with 5 years of age and 50000 mileage: {predicted_price[0]}')