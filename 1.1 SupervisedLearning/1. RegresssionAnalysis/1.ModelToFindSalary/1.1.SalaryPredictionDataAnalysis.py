# Pandas is a powerful data manipulation library in Python, widely used for data analysis tasks.
import pandas as pd

# Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
import matplotlib.pyplot as plt

# NumPy is a fundamental package for scientific computing in Python, providing support for arrays and mathematical functions.
import numpy as np

data_frame = pd.read_csv('salary_data.csv')

# Display the first few rows of the DataFrame to understand its structure and contents.
# print('First few rows of the DataFrame:', data_frame.head())

# Display the column names of the DataFrame to understand the features available in the dataset.
print('Column names in the DataFrame:', data_frame.columns)

# Display the shape of the DataFrame to understand the number of rows and columns in the dataset.
# print('Shape of the DataFrame:', data_frame.shape)

# Display the data types of each column in the DataFrame to understand the nature of the data.
# print('Data types of each column:', data_frame.dtypes)

# Display summary statistics of the DataFrame to get insights into the distribution and central tendencies of the data.
# print('Summary statistics of the DataFrame:', data_frame.describe())

# Display information about the DataFrame, including non-null counts and memory usage.
# print('Information about the DataFrame:')
# data_frame.info()

# Check for missing values in the DataFrame to identify any gaps in the data.
# print('Missing values in each column:', data_frame.isnull().sum())

# Check for duplicate rows in the DataFrame to ensure data integrity.
# print('Number of duplicate rows in the DataFrame:', data_frame.duplicated().sum())

# Display the correlation matrix to understand the relationships between different features in the dataset.
# print('Correlation matrix of the DataFrame:', data_frame.corr())

correlation = data_frame['Experience'].corr(data_frame['Salary'])
# print('Correlation between Experience and Salary:', correlation)

covariance = np.cov(data_frame['Experience'], data_frame['Salary'])
# print('Covariance between Experience and Salary:', covariance)

# print('Mean Salary:', data_frame['Salary'].mean())
# print('Median Salary:', data_frame['Salary'].median())
# print('Mode Salary:', data_frame['Salary'].mode()[0])

# Create a scatter plot to visualize the relationship between years of experience and salary.
# plt.scatter(data_frame['Experience'], data_frame['Salary'])
# plt.title('Experience vs Salary')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.grid(True)
# plt.show()

# Create a histogram to visualize the distribution of salaries in the dataset.
# data_frame["Salary"].hist(bins=10, edgecolor='black')
# plt.title('Salary Distribution')
# plt.xlabel('Salary')
# plt.ylabel('Frequency')
# plt.grid(False)
# plt.show()