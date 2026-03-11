import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Title
st.title("🚗 Used Car Price Prediction")

# Load Dataset
df = pd.read_csv("car data.csv")

# Feature Engineering
current_year = 2026
df["Car_Age"] = current_year - df["Year"]

# Drop unnecessary columns
df.drop(["Year", "Car_Name"], axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()
df["Fuel_Type"] = le.fit_transform(df["Fuel_Type"])
df["Seller_Type"] = le.fit_transform(df["Seller_Type"])
df["Transmission"] = le.fit_transform(df["Transmission"])

# Define Features and Target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Evaluate Model
rf_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
r2 = r2_score(y_test, rf_pred)

st.subheader("Model Performance")
st.write("RMSE:", rmse)
st.write("R² Score:", r2)

# ---------------------------
# User Input Section
# ---------------------------

st.subheader("Enter Car Details")

present_price = st.number_input("Present Price of Car (Lakhs)")
kms_driven = st.number_input("Kilometers Driven")

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual"]
)

owner = st.number_input("Number of Previous Owners", min_value=0)

car_age = st.number_input("Car Age (years)")

selling_type = st.selectbox(
    "Selling Type",
    ["Dealer", "Individual"]
)

# Convert categorical to numeric
fuel_type = 0 if fuel_type == "Petrol" else 1
seller_type = 0 if seller_type == "Dealer" else 1
transmission = 0 if transmission == "Automatic" else 1
selling_type = 0 if selling_type == "Dealer" else 1

# Prediction Button
if st.button("Predict Price"):

    new_car = pd.DataFrame({
        'Present_Price':[present_price],
        'Kms_Driven':[kms_driven],
        'Fuel_Type':[fuel_type],
        'Seller_Type':[seller_type],
        'Transmission':[transmission],
        'Owner':[owner],
        'Car_Age':[car_age],
        'Selling_type':[selling_type]
    })

    predicted_price = rf.predict(new_car)

    st.success(f"Predicted Price: ₹ {predicted_price[0]:.2f} Lakhs")