# STOCK-FORCASTING
Stock Market Data Analysis, Outlier Detection, and Predictive Modeling
Project Title: Stock Market Data Analysis, Outlier Detection, and Predictive Modeling
Description
This project provides a comprehensive framework for analyzing stock market data, including loading, preprocessing, outlier detection, feature selection, and predictive modeling using Linear Regression. The project employs various Python libraries such as pandas, numpy, matplotlib, seaborn, scikit-learn, and mlxtend to conduct exploratory data analysis (EDA), visualize data, detect outliers, and build a predictive model to forecast stock prices.

Features
Data Loading and Preprocessing: Load stock market data from a CSV file and perform data cleaning, normalization, and handling of missing values.
Outlier Detection and Removal: Identify and remove outliers from the dataset using the Interquartile Range (IQR) method.
Correlation Analysis: Analyze and visualize the correlation between different features using a heatmap.
Data Splitting: Split the data into training and testing sets for model validation.
Feature Selection: Select the most relevant features for predictive modeling using Sequential Feature Selection (SFS) with a Linear Regression model.
Predictive Modeling: Train a Linear Regression model to predict stock prices and evaluate its performance using metrics like Mean Squared Error (MSE) and R-squared Score (R²).
Cross-Validation: Perform cross-validation to assess the model's robustness and generalization capabilities.
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/stock-market-analysis.git
Navigate to the project directory:
bash
Copy code
cd stock-market-analysis
Install the required packages:
bash
Copy code
pip install -r requirements.txt
Usage
Data Loading:

Replace the df=pd.read_csv line with the actual path to your CSV file containing stock market data.
The dataset should include columns like Date, Open, High, Low, Close, Adj Close, and Volume.
Outlier Detection and Removal:

The project uses a box plot to visualize outliers in the dataset.
The IQR method is applied to filter out outliers, and a clean dataset without outliers (df1) is created.
Correlation Analysis:

The project calculates the correlation matrix for the cleaned dataset and visualizes it using a heatmap to understand the relationships between different features.
Data Splitting:

The cleaned dataset is split into training and testing sets, with Close being the target variable for prediction.
Feature Selection:

Sequential Feature Selection (SFS) is used to identify the most significant features for the prediction model. In this case, the features selected are Open, High, and Low.
Model Training and Validation:

A Linear Regression model is trained on the selected features. The model's performance is evaluated using MSE and R² scores.
Cross-validation is performed to validate the model's stability and generalization.
Example Code:

python
Copy code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Load and preprocess data
df = pd.read_csv('your-stock-data.csv')
df.drop_duplicates(inplace=True)
df.fillna(method='ffill', inplace=True)

# Detect and remove outliers
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
IQR = q3 - q1
lowerlimit = q1 - 1.5 * IQR
upperlimit = q3 + 1.5 * IQR
filter = (df >= lowerlimit) & (df <= upperlimit)
df1 = df[filter.all(axis=1)]

# Correlation analysis
corr_matrix = df1.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlations between all features")
plt.show()

# Data splitting
X = df1[['Open', 'High', 'Low']]
y = df1['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection and model training
model = LinearRegression()
sfs = SFS(model, k_features='best', forward=True, floating=False, scoring='neg_mean_squared_error', cv=5)
sfs.fit(X_train, y_train)
selected_features = df1.columns[list(sfs.k_feature_idx_)]
print("Selected Features:", selected_features)

# Model validation
model.fit(X_train[selected_features], y_train)
y_pred = model.predict(X_test[selected_features])
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Cross-validation
cv_scores = cross_val_score(model, X_train[selected_features], y_train, cv=5, scoring='r2')
print("Cross-validation R-squared scores:", cv_scores)
print("Mean R-squared score from cross-validation:", np.mean(cv_scores))
Requirements
numpy
pandas
matplotlib
seaborn
scikit-learn
mlxtend
Install all required packages using:

bash
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend
Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request to enhance the project.

License
This project is licensed under the MIT License.

