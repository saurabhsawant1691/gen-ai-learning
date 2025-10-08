import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('salary_data.csv')

x = df.drop('Salary', axis=1)
y = df['Salary']

column_transformer = ColumnTransformer(
  transformers = [
    ('onehot', OneHotEncoder(sparse_output=True, drop='first'), ['Title'])
  ],
  remainder='passthrough'
)

transformed_values = column_transformer.fit_transform(x)

transformed_features = pd.DataFrame(transformed_values, columns=column_transformer.get_feature_names_out())

# print(transformed_features)

model = LinearRegression()
model.fit(transformed_features, y)

data_to_predict = pd.DataFrame([['Project Manager',2]], columns=['Title', 'Experience'])

# print(data_to_predict)

new_data_transformed = column_transformer.transform(data_to_predict)

# print(new_data_transformed)

data_frame_to_predict = pd.DataFrame(new_data_transformed, columns=column_transformer.get_feature_names_out())

y_prod = model.predict(data_frame_to_predict)
print(y_prod)