import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv('salary_data.csv')

plt.scatter(df['Experience'], df['Salary'])
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.show()

x = df.drop('Salary', axis=1)
y = df['Salary']

model = LinearRegression()
model.fit(x, y)

re_predict_all_x = model.predict(x)

output = model.predict([[17.5]])
print(output)

plt.scatter(df['Experience'], df['Salary'], label='Observed Data')
plt.scatter(df['Experience'], re_predict_all_x, color='red', label='Predicted Data')
plt.plot(df['Experience'], re_predict_all_x, color='red', label='Best Fit Line')
plt.legend()
plt.show()

score = model.score(x, y)
print(score)


