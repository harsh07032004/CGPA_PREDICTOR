import pandas as pd
import numpy as np
import joblib  # For saving the trained model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load the encoded dataset
df_encoded = pd.read_csv("df_encoded.csv")  # Ensure df_encoded.csv exists in the same directory

# Step 2: Define features (X) and target variable (y)
X = df_encoded.drop(columns=['CGPA'])  # Independent variables
y = df_encoded['CGPA']  # Target variable

# Step 3: Splitting into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Linear Regression Model
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)

# Step 6: Make Predictions
y_pred_lr = model_lr.predict(X_test_scaled)

# Step 7: Evaluate Model Performance
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

# Print error metrics
print(f"Mean Absolute Error (MAE): {mae_lr}")
print(f"Mean Squared Error (MSE): {mse_lr}")
print(f"Root Mean Squared Error (RMSE): {rmse_lr}")

# Step 8: Save the trained model and scaler
joblib.dump(model_lr, "linear_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")
