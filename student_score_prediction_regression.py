# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

# 1. Load dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

# 2. Select only relevant columns
df = df[['Hours_Studied', 'Exam_Score']]

# 3. Data Cleaning
print("Missing values before cleaning:\n", df.isnull().sum())

# Drop rows with missing values 
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# 4. Remove Outliers Using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.6)
IQR = Q3 - Q1

# Keep only data within the IQR range
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# 5. Show stats after cleaning and outlier removal
print("\nData Summary after cleaning and outlier removal:")
print(df.describe())

# 6. Basic Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Hours_Studied', y='Exam_Score')
plt.title("Study Hours vs Exam Score (Outliers Removed)")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

# Correlation heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 7. Split Data into Training and Testing
X = df[['Hours_Studied']]
y = df['Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)

### 8. Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)

### 9. Polynomial Regression (degree = 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_pred = poly_model.predict(X_test_poly)

### 10. Visualize Both Models
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, label='Actual', color='blue')
plt.plot(X_test, lin_pred, label='Linear Regression', color='red')
plt.scatter(X_test, poly_pred, label='Polynomial Regression (deg=2)', color='green', marker='x')
plt.title("Actual vs Predicted Exam Scores (Linear vs Polynomial deg=2)")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True)
plt.show()

### 11. Evaluation Comparison
print("\nModel Evaluation Comparison:")

# Linear metrics
lin_mae = mean_absolute_error(y_test, lin_pred)
lin_mse = mean_squared_error(y_test, lin_pred)
lin_r2 = r2_score(y_test, lin_pred)

print("Linear Regression:")
print(f"  MAE: {lin_mae:.2f}")
print(f"  MSE: {lin_mse:.2f}")
print(f"  R²:  {lin_r2:.2f}")

# Polynomial metrics
poly_mae = mean_absolute_error(y_test, poly_pred)
poly_mse = mean_squared_error(y_test, poly_pred)
poly_r2 = r2_score(y_test, poly_pred)

print("Polynomial Regression (degree=2):")
print(f"  MAE: {poly_mae:.2f}")
print(f"  MSE: {poly_mse:.2f}")
print(f"  R²:  {poly_r2:.2f}")
