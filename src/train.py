import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data/housing.csv")

# Remove missing values
data = data.dropna()

# Convert categorical column to numeric
data = pd.get_dummies(data)

# Target column
target = "median_house_value"

# Features and label
X = data.drop(columns=[target])
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

# Print metrics for GitHub pipeline
print("Dataset Size:", len(X_train))
print("RMSE:", rmse)
print("R2:", r2)