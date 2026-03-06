import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/housing.csv")

# Example feature selection (adjust if needed)
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, pred, squared=False)
r2 = r2_score(y_test, pred)

dataset_size = len(X_train)

rmse = mean_squared_error(y_test, pred, squared=False)
r2 = r2_score(y_test, pred)

print("Dataset Size:", len(X_train))
print("RMSE:", rmse)
print("R2:", r2)