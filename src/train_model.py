import pandas as pd
import numpy as np
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Load dataset
df = pd.read_csv("/Users/manideepak/Desktop/PycharmProjects/Retail_Demad_Project/data/archive-2/train.csv")

# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# Feature engineering
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['week'] = df['Date'].dt.isocalendar().week.astype(int)

# Handle missing values
df.ffill(inplace=True)

# Features & target
X = df[['Store', 'Dept', 'year', 'month', 'week']]
y = df['Weekly_Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=50)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

rmse = root_mean_squared_error(y_test, pred)
print("RMSE:", rmse)



# MLflow tracking
# mlflow.start_run()
# mlflow.log_metric("rmse", rmse)
# mlflow.end_run()

# Save model
joblib.dump(model, "/Users/manideepak/Desktop/PycharmProjects/Retail_Demand_Project/models/model.pkl")

print("Model training complete!")
