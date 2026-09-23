import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# ==========================================
# 1. LOAD DATASET
# ==========================================
data = pd.read_csv("cardekho_dataset.csv")
print("Dataset loaded successfully!")
print("Total rows:", len(data))
# ==========================================
# 2. REMOVE UNNECESSARY COLUMN
# ==========================================
data = data.drop("Unnamed: 0", axis=1)
# ==========================================
# 3. SEPARATE INPUT AND OUTPUT
# ==========================================
X = data.drop("selling_price", axis=1)
y = data["selling_price"]
# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("Training data:", X_train.shape[0])
print("Testing data :", X_test.shape[0])
# ==========================================
# 5. CATEGORICAL COLUMNS
# ==========================================
categorical_columns = [
    "car_name",
    "brand",
    "model",
    "seller_type",
    "fuel_type",
    "transmission_type"
]
# ==========================================
# 6. CONVERT TEXT INTO NUMBERS
# ==========================================
encoder = ColumnTransformer(
    transformers=[
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
# ==========================================
# 7. RANDOM FOREST MODEL
# ==========================================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
# ==========================================
# 8. TRAIN MODEL
# ==========================================
print("\nTraining Random Forest...")
model.fit(X_train, y_train)
print("Training completed!")
# ==========================================
# 9. TEST PREDICTIONS
# ==========================================
predictions = model.predict(X_test)
# ==========================================
# 10. MODEL EVALUATION
# ==========================================
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)
print("\n========== MODEL RESULTS ==========")
print("R² Score :", round(r2, 4))
print("R² %     :", round(r2 * 100, 2), "%")
print("MAE      : ₹", round(mae, 2))
print("RMSE     : ₹", round(rmse, 2))
# ==========================================
# 11. USER INPUT
# ==========================================
print("\n========================================")
print("       CAR PRICE PREDICTION")
print("========================================")
print("\nEnter the details of your car:")
car_name = input("Car Name: ")
brand = input("Brand: ")
model_name = input("Model: ")
vehicle_age = int(input("Vehicle Age (years): "))
km_driven = int(input("Kilometers Driven: "))
seller_type = input("Seller Type (Individual/Dealer/Trustmark Dealer): ")
fuel_type = input("Fuel Type (Petrol/Diesel/CNG/LPG/Electric): ")
transmission_type = input("Transmission (Manual/Automatic): ")
mileage = float(input("Mileage (km/l): "))
engine = int(input("Engine (CC): "))
max_power = float(input("Max Power (bhp): "))
seats = int(input("Number of Seats: "))
# ==========================================
# 12. CREATE NEW CAR DATA
# ==========================================
new_car = pd.DataFrame({
    "car_name": [car_name],
    "brand": [brand],
    "model": [model_name],
    "vehicle_age": [vehicle_age],
    "km_driven": [km_driven],
    "seller_type": [seller_type],
    "fuel_type": [fuel_type],
    "transmission_type": [transmission_type],
    "mileage": [mileage],
    "engine": [engine],
    "max_power": [max_power],
    "seats": [seats]
})
# ==========================================
# 13. ENCODE NEW CAR
# ==========================================
new_car_encoded = encoder.transform(new_car)
# ==========================================
# 14. PREDICT PRICE
# ==========================================
predicted_price = model.predict(new_car_encoded)
# ==========================================
# 15. DISPLAY PREDICTED PRICE
# ==========================================
print("\n========================================")
print("       PREDICTED CAR PRICE")
print("========================================")
print(
    "Estimated Selling Price: ₹",
    round(predicted_price[0], 2)
)
print("========================================")