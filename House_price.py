import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: App Title
st.title("Dynamic House Price Prediction App")
st.write("This app uses dummy data and allows you to add new data for dynamic model training.")

# Step 2: Initial Dummy Data
initial_data = {
    "Size (sqft)": [1500, 1800, 2000, 2300, 2500, 2700, 3000, 3200],
    "Bedrooms": [3, 4, 3, 4, 5, 4, 5, 6],
    "Price ($)": [300000, 400000, 450000, 500000, 600000, 650000, 700000, 750000],
}
df = pd.DataFrame(initial_data)

# Step 3: Allow User to Add New Data
st.write("### Add New Data")
with st.form("add_data_form"):
    size = st.number_input("Enter size (sqft):", min_value=500, max_value=5000, step=100)
    bedrooms = st.number_input("Enter number of bedrooms:", min_value=1, max_value=10, step=1)
    price = st.number_input("Enter price ($):", min_value=50000, max_value=1000000, step=1000)
    submitted = st.form_submit_button("Add Data")

# Step 4: Update Dataset
if submitted:
    new_row = {"Size (sqft)": size, "Bedrooms": bedrooms, "Price ($)": price}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    st.success("New data added successfully!")

st.write("### Current Dataset")
st.write(df)

# Step 5: Train the Model Dynamically
X = df[["Size (sqft)", "Bedrooms"]]
y = df["Price ($)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

st.write("### Model Training Results")
st.write(f"Intercept: {model.intercept_:.2f}")
st.write(f"Coefficients: Size = {model.coef_[0]:.2f}, Bedrooms = {model.coef_[1]:.2f}")

# Step 6: Visualize Predictions
predictions = model.predict(X_test)

st.write("### Actual vs Predicted Prices (Test Data)")
fig, ax = plt.subplots()
ax.scatter(y_test, predictions, color="blue", label="Predicted")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Ideal")
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.legend()
st.pyplot(fig)

# Step 7: Make Predictions
st.write("### Make Your Own Prediction")
size_input = st.number_input("Enter size (sqft):", min_value=500, max_value=5000, step=100)
bedrooms_input = st.number_input("Enter number of bedrooms:", min_value=1, max_value=10, step=1)
if st.button("Predict Price"):
    predicted_price = model.predict([[size_input, bedrooms_input]])
    st.write(f"### Predicted Price: ${predicted_price[0]:,.2f}")
