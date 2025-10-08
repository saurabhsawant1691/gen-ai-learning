import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('spending_data.csv')

age_spending_correlation = df['Age'].corr(df['Spendings'])
salary_spending_correlation = df['Salary'].corr(df['Spendings'])

print(f'Age-Spending Correlation: {age_spending_correlation}')
print(f'Salary-Spending Correlation: {salary_spending_correlation}')

# plt.scatter(df['Age'], df['Spendings'])
# plt.xlabel('Age')
# plt.ylabel('Spendings')
# plt.title('Age vs Spendings')
# plt.show()

# plt.scatter(df['Salary'], df['Spendings'])
# plt.xlabel('Salary')
# plt.ylabel('Spendings')
# plt.title('Salary vs Spendings')
# plt.show()

x = df.drop('Spendings', axis = 1)
y = df['Spendings']

model = LinearRegression()
model.fit(x, y)

output = model.predict(pd.DataFrame([[53, 115541]], columns = ['Age', 'Salary']))
print(output)
