#Importing the Dependencies
import streamlit as st
#Data handling and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Data preprocessing and data transformation
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
#Model selection and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

st.title("Housing Price Prediction")

st.write("Dữ liệu train ban đầu: ")
st.write(df_train.head())

#Separate the features and the target variable
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice'] # Target variable

#Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4. Drop the unwanted columns
#Check the percentage of missing values on each columns
missing_values = X_train.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values_percentage = (missing_values / len(X_train)) * 100
st.write(missing_values_percentage)

st.write("Columns with missing values when visualizing: ")

#Visualize all columns that missing at percentages
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values_percentage.index, y=missing_values_percentage.values, palette='viridis')
plt.title('Missing Values Percentage')
plt.xlabel('Columns')
plt.ylabel('Percentage')
plt.xticks(rotation=90)
st.pyplot(plt)

#Check and drop columns that have many missing data
columns_to_drop = missing_values_percentage[missing_values_percentage > 30].index
X_train.drop(columns=columns_to_drop, inplace=True)
X_test.drop(columns=columns_to_drop, inplace=True)

#check columns to fill missing
columns_to_fill = missing_values_percentage[missing_values_percentage < 30].index

#fill missing data
for column in columns_to_fill:
  if X_train[column].dtype == 'float64' or X_train[column].dtype == 'int64':
    X_train[column].fillna(X_train[column].mean(), inplace=True)
    X_test[column].fillna(X_test[column].mean(), inplace=True)
  else:
    X_train[column].fillna(X_train[column].mode()[0], inplace=True)
    X_test[column].fillna(X_test[column].mode()[0], inplace=True)

st.write(X_train.hist(figsize=(21,15)))

st.header("Before preprocessing data")
st.write("Here is the heatmap before filling missing data and dropping columns with many missing data: ")

#Visualize the correlation with heatmap
corrMatrix = df_train.select_dtypes(exclude=['object']).corr()
plt.subplots(figsize=(60, 20))
sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features and target')
st.pyplot(plt)
#Nhận xét: Các cột có giá trị tương quan với cột SalePrice dưới 0.05 sẽ được loại bỏ vì không ảnh hưởng nhiều đến giá nhà.
st.write("Nhận xét: Các cột có giá trị tương quan với cột SalePrice dưới 0.05 sẽ được loại bỏ vì không ảnh hưởng nhiều đến giá nhà.")

st.write("Columns with correlation less than 0.05: ")
#check and drop features with a very low correlation
columns_to_drop_corr = corrMatrix[corrMatrix['SalePrice'] < 0.05].index
X_train.drop(columns=columns_to_drop_corr, inplace=True)
X_test.drop(columns=columns_to_drop_corr, inplace=True)
st.write(columns_to_drop_corr)

num_data = X_train.select_dtypes(include=np.number).columns.tolist()
cat_data = X_train.select_dtypes(exclude=np.number).columns.tolist()

st.write("Numerical data: ", num_data)
st.write("Categorical data: ", cat_data)

#Based on heatmap above we remove these columns
num_data.remove('GarageArea')
num_data.remove('1stFlrSF')
num_data.remove('GrLivArea')
num_data.remove('BsmtFullBath')
num_data.remove('FullBath')
num_data.remove('HalfBath')
num_data.remove('TotRmsAbvGrd')
num_data.remove('GarageYrBlt')

st.header("After preprocessing data")
st.write("Correlation after removing columns: ")
corr = X_train[num_data].corr()
plt.subplots(1, 1, figsize=(25, 25))
sns.heatmap(data=corr, cmap=sns.diverging_palette(20, 220, n=200), annot=True)
plt.title('Correlation between features')
st.pyplot(plt)

st.write("Data after preprocessing of X_train: ")
st.write(X_train.head())

st.header("Outliers EDA")

#Set up the figure size and DPI
plt.figure(figsize=(15,3), dpi=150)

#Create the boxplot
sns.boxplot(x=y)

plt.xlabel('SalePrice')
plt.title('SalePrice Distribution')
st.pyplot(plt)

st.write("Nhận xét:")
st.write("+ Phần lớn giá nhà tập trung trong khoảng từ khoảng 100,000 đến 250,000 (dựa trên hộp của boxplot).")
st.write("+ Đường mediane (đường nằm ngang trong hộp) cho thấy giá nhà có xu hướng hơi nghiêng về phía thấp hơn trong nhóm này.")
st.write("+ Phân phối giá nhà dường như bị lệch phải (right-skewed), điều này phổ biến trong dữ liệu giá nhà vì thường có một số ít căn nhà giá rất cao (nhà hạng sang) trong khi đa số các căn nhà có giá trung bình.")

st.write("Cho nên giải pháp hay nhất là cần phải xử lý outliers")

# Giới hạn outliers bằng IQR
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = y[(y < lower_bound) | (y > upper_bound)]
st.write("Số lượng outliers: ", len(outliers))

st.header("Visualize EDA:")

n_features = len(num_data)
n_cols = 3
n_rows = np.ceil(n_features / n_cols).astype(int)

#Create the subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 5 * n_rows))

#Flatten the axis array for easy iteration
axes = axes.flatten()

#Loop through each feature and plot a boxplot
for i, feature in enumerate(num_data):
    X_train[[feature]].boxplot(ax=axes[i])
    axes[i].set_title(feature)

#Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

#Adjust the layout and show the plot
plt.tight_layout()
st.pyplot(plt)

st.header("Create a Pipeline Model:")
num_preprocessor = make_pipeline(SimpleImputer(strategy='median'), MinMaxScaler())
cat_preprocessor = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))

complete_pipeline = ColumnTransformer([
    ('num_preprocessor', num_preprocessor, num_data),
    ('cat_preprocessor', cat_preprocessor, cat_data)
])
st.write(complete_pipeline)

import statsmodels.api as sm

st.header("Model Selection and Evaluation:")


#Build the Linear Regression Model - Multiple Linear Regression
class LinearRegression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, x, y):
        #no of training examples, no of features
        self.m, self.n = x.shape
        #weight initialization
        self.W = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y
        #gradient descent learning
        for i in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        Y_pred = self.predict(self.x)
        #calculate gradients
        dW = -(2 * (self.x.T).dot(self.y - Y_pred)) / self.m
        db = -2 * np.sum(self.y - Y_pred) / self.m
        #update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict(self, x):
        return x.dot(self.W) + self.b

def prediction(model):
    pipeline = Pipeline([
        ('preprocessor', complete_pipeline),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    st.write(f"{model}")
    st.write(f"RMSE is {np.sqrt(mean_squared_error(y_test, y_pred))}")
    st.write(f"MAE is {mean_absolute_error(y_test, y_pred)}")
    st.write(f"MSE is {mean_squared_error(y_test, y_pred)}")
    st.write(f"R2 is {r2_score(y_test, y_pred)}")


st.write("**Multiple Linear Regression Model**")
st.write(prediction(LinearRegression(0.01, 1000)))

st.write("**Ridge Regression Model**")
from sklearn.linear_model import Ridge
prediction(Ridge())

# Draw a suitable plot to visualize the relationship between the actual and predicted values.
def plot_actual_vs_predicted(model):
    pipeline = Pipeline([
        ('preprocessor', complete_pipeline),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    st.pyplot(plt)

st.write("**Actual vs Predicted for Multiple Linear Regression Model**")
plot_actual_vs_predicted(LinearRegression(0.01, 1000))

st.write("**Actual vs Predicted for Ridge Regression Model**")
plot_actual_vs_predicted(Ridge())
