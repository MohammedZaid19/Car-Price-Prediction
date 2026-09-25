# 🚗 Car Price Prediction Using Machine Learning

## 📌 Project Overview

Car Price Prediction is a machine learning project that predicts the estimated selling price of a used car based on its features. The project uses the CarDekho Used Car Dataset and a Random Forest Regression model to learn patterns from existing car data and predict prices for new car details.

## 🎯 Objective

The main objective of this project is to develop a machine learning system that can estimate the selling price of a used car using its specifications and condition-related features.

## 📊 Dataset

The project uses the **CarDekho Used Car Dataset**.

- Total Records: **15,411**
- Original Columns: **14**
- Target Variable: `selling_price`
- Dataset Format: CSV
- Missing Values: None

### Features Used

- Car Name
- Brand
- Model
- Vehicle Age
- Kilometers Driven
- Seller Type
- Fuel Type
- Transmission Type
- Mileage
- Engine
- Max Power
- Seats

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

## 🤖 Machine Learning Algorithm

### Random Forest Regression

Random Forest Regression is used to predict the selling price of used cars. It combines multiple decision trees to produce a more reliable prediction.

## ⚙️ Methodology

1. Load the CarDekho dataset.
2. Check the dataset structure and missing values.
3. Remove the unnecessary index column.
4. Separate input features and target variable.
5. Split the dataset into 80% training and 20% testing data.
6. Identify categorical and numerical features.
7. Apply One-Hot Encoding to categorical features.
8. Train the Random Forest Regression model.
9. Predict prices using unseen test data.
10. Evaluate the model using R² Score, MAE, and RMSE.
11. Accept new car details from the user.
12. Predict the estimated selling price of the entered car.

## 📈 Model Performance

The Random Forest model was evaluated on the unseen 20% testing data.

| Metric | Result |
|--------|--------|
| R² Score | **93.87%** |
| MAE | **98,194.53** |
| RMSE | **214,813.84** |

## 💻 User Input Prediction

The project allows the user to enter details of a new car, such as:

- Car Name
- Brand
- Model
- Vehicle Age
- Kilometers Driven
- Seller Type
- Fuel Type
- Transmission
- Mileage
- Engine
- Max Power
- Number of Seats

The trained model then provides an **estimated selling price**.

## 📁 Project Structure

```text
Car-Price-Prediction/
│
├── cardekho_dataset.csv
├── car_price_prediction.py
└── README.md
🚀 How to Run
1. Install Required Libraries
pip install pandas numpy scikit-learn
2. Keep the Dataset

Place cardekho_dataset.csv in the same folder as the Python file.

3. Run the Project
python car_price_prediction.py
4. Enter Car Details

Enter the required car information when prompted to get the estimated selling price.

📌 Applications
Used-car price estimation
Price comparison for buyers
Price estimation for sellers
Used-car dealership support
Data-driven car valuation
Integration with car-selling websites or applications
