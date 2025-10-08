import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv')

print('Null value count', df.isnull().sum())

missing_dataframe = df[df.isnull().any(axis=1)]

print('Rows with missing values', missing_dataframe)

df = df.dropna()

print(df)

x = df.drop('Salary', axis=1)
y = df['Salary']

model = LinearRegression()

model.fit(x, y)

result = model.predict(pd.DataFrame([4], columns=['Experience']))

print(result)

predicted_rows = []

for ids, row in missing_dataframe.iterrows():
  experience = row['Experience']
  input_df = pd.DataFrame([[experience]], columns=['Experience'])
  predicted_salary = model.predict(input_df)
  predicted_rows.append({'Experience': experience, 'Salary': predicted_salary})

predicted_pd = pd.DataFrame(predicted_rows)

df = pd.concat([df, predicted_pd], ignore_index=True)

print('Final DataFrame: ',df)

x_new = df.drop('Salary', axis=1)
y_new = df['Salary']

model.fit(x_new, y_new)

final_output = model.predict(pd.DataFrame([12], columns=['Experience']))

print('Final Salary Prediction:', final_output)