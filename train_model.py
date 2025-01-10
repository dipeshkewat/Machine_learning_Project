import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the cleaned dataset
df = pd.read_csv('cleaned_cars.csv')

X = df[['Age', 'Mileage', 'Horsepower', 'Cylinders']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

joblib.dump(model, 'car_price_model.pkl')
print("Model saved to 'car_price_model.pkl'")
