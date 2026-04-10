# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1️⃣ Train Model
# -----------------------------
@st.cache_resource
def train_model():
    data = pd.read_csv("Food_Delivery_Times.csv")
    data['Weather'] = data['Weather'].fillna(data['Weather'].mode()[0])
    data['Traffic_Level'] = data['Traffic_Level'].fillna(data['Traffic_Level'].mode()[0])
    data['Time_of_Day'] = data['Time_of_Day'].fillna(data['Time_of_Day'].mode()[0])
    data['Courier_Experience_yrs'] = data['Courier_Experience_yrs'].fillna(data['Courier_Experience_yrs'].median())
    
    # One-hot encoding
    data = pd.get_dummies(data, columns=['Weather','Traffic_Level','Time_of_Day','Vehicle_Type'])
    
    X = data.drop(columns=['Order_ID','Delivery_Time_min'])
    y = data['Delivery_Time_min']
    
    model = GradientBoostingRegressor()
    model.fit(X, y)
    
    # 🔥 Calculate RMSE for range
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    return model, X.columns, rmse


model, feature_columns, rmse = train_model()

# -----------------------------
# 2️⃣ UI
# -----------------------------
st.title("Food Delivery Time Predictor 🍔🛵")
st.write("Predict delivery time as a realistic range instead of exact value.")

# Inputs
distance = st.number_input("Distance (km)", 0.1, 50.0, 5.0)
courier_exp = st.number_input("Courier Experience (yrs)", 0, 20, 2)
prep_time = st.number_input("Preparation Time (min)", 1, 60, 10)

weather = st.selectbox("Weather", ["Clear", "Rainy", "Snowy", "Foggy", "Windy"])
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
vehicle_type = st.selectbox("Vehicle Type", ["Bike", "Scooter", "Car", "Motorcycle"])

# -----------------------------
# 3️⃣ Prepare Input
# -----------------------------
input_dict = {
    'Distance_km': distance,
    'Courier_Experience_yrs': courier_exp,
    'Preparation_Time_min': prep_time
}

# One-hot encoding
for col in feature_columns:
    if col.startswith("Weather_"):
        input_dict[col] = 1 if col == "Weather_" + weather else 0
    elif col.startswith("Traffic_Level_"):
        input_dict[col] = 1 if col == "Traffic_Level_" + traffic else 0
    elif col.startswith("Time_of_Day_"):
        input_dict[col] = 1 if col == "Time_of_Day_" + time_of_day else 0
    elif col.startswith("Vehicle_Type_"):
        input_dict[col] = 1 if col == "Vehicle_Type_" + vehicle_type else 0

input_df = pd.DataFrame([input_dict])

# 🔥 Ensure same columns
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# -----------------------------
# 4️⃣ Prediction (Range)
# -----------------------------
if st.button("Predict Delivery Time"):
    with st.spinner("Predicting..."):
    prediction = model.predict(input_df)[0]
    
    lower = max(0, prediction - rmse)
    upper = prediction + rmse
    
    # Show range
    st.success(f"Estimated Delivery Time: {lower:.0f} - {upper:.0f} minutes")
    

    
    # Extra interpretation
    if prediction < 20:
        st.info("🚀 Fast delivery expected!")
    elif prediction < 40:
        st.warning("⏳ Moderate delivery time.")
    else:
        st.error("🐢 Delivery might be delayed.")
