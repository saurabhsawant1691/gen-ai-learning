import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Advertising.csv')

print(df.columns);
df.info()

corr_tv_sales = df['TV'].corr(df['sales'])
corr_radio_sales = df['radio'].corr(df['sales'])
corr_newspaper_sales = df['newspaper'].corr(df['sales'])

print(f'Correlation between TV and Sales: {corr_tv_sales}')
print(f'Correlation between Radio and Sales: {corr_radio_sales}')
print(f'Correlation between Newspaper and Sales: {corr_newspaper_sales}')

plt.scatter(df['TV'], df['sales'])
plt.xlabel('TV Advertisement')
plt.ylabel('Sales')
plt.title('TV Advertisement vs Sales')
plt.show()

plt.scatter(df['radio'], df['sales'])
plt.xlabel('Radio Advertisement')
plt.ylabel('Sales')
plt.title('Radio Advertisement vs Sales')
plt.show()

plt.scatter(df['newspaper'], df['sales'])
plt.xlabel('Newspaper Advertisement')
plt.ylabel('Sales')
plt.title('Newspaper Advertisement vs Sales')
plt.show()

model = LinearRegression()

# Drop the 'sales' and 'newspaper' column to create the feature set
x = df.drop(['sales', 'newspaper'], axis=1)
y = df['sales']

print(x)
print(y)

model.fit(x, y)

input_data = pd.DataFrame([[50.23, 30.45]], columns=['TV', 'radio'])
print(input_data)

predicted_sales = model.predict(input_data)

print(f'Predicted sales for a company with TV advertisement of 50.23 and Radio advertisement of 30.45: {predicted_sales[0]}')