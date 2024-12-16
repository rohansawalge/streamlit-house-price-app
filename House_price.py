import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: App Title
st.title("House Price Prediction App")
st.write("This app uses dummy data to predict house prices based on size and number of bedrooms.")

# Step 2: Create Dummy Data
data = {
    "Size (sqft)": [1500, 1800, 2000, 2300, 2500, 2700, 3000, 3200],
    "Bedrooms": [3, 4, 3, 4, 5, 4, 5, 6],
    "Price ($)": [300000, 400000, 450000, 500000, 600000, 650000, 700000, 750000],
}
df = pd.DataFrame(data)

st.write("### Dummy Dataset")
st.write(df)

# Step 3: Prepare Data
X = df[["Size (sqft)", "Bedrooms"]]
y = df["Price ($)"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Show model coefficients
st.write("### Model Training Results")
st.write(f"Intercept: {model.intercept_:.2f}")
st.write(f"Coefficients: Size = {model.coef_[0]:.2f}, Bedrooms = {model.coef_[1]:.2f}")

# Step 5: Visualize Predictions
predictions = model.predict(X_test)

st.write("### Actual vs Predicted Prices (Test Data)")
fig, ax = plt.subplots()
ax.scatter(y_test, predictions, color="blue", label="Predicted")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Ideal")
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.legend()
st.pyplot(fig)

# Step 6: Make Predictions
st.write("### Make Your Own Prediction")
size = st.number_input("Enter size (sqft):", min_value=500, max_value=5000, step=100)
bedrooms = st.number_input("Enter number of bedrooms:", min_value=1, max_value=10, step=1)
if st.button("Predict Price"):
    predicted_price = model.predict([[size, bedrooms]])
    st.write(f"### Predicted Price: ${predicted_price[0]:,.2f}")
