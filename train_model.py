import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the cleaned dataset
df = pd.read_csv('cleaned_cars.csv')

# Features and target
X = df[['Age', 'Mileage', 'Horsepower', 'Cylinders']]
y = df['Price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Save the trained model
joblib.dump(model, 'car_price_model.pkl')
print("Model saved to 'car_price_model.pkl'")
