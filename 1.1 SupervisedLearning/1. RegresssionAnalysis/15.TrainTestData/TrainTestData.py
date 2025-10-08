import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv('spending_data.csv')

X = data.drop("Spendings", axis = 1)
y = data["Spendings"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

model = LinearRegression()
model.fit(X_train, y_train)

for x_test in X_test.values:
  print(f'Input: {x_test}, Prediction: {model.predict([x_test])}')

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f'Train Score: {train_score}')
print(f'Test Score: {test_score}')