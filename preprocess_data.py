import pandas as pd

# Load dataset
df = pd.read_csv('cars.csv')

# Select relevant columns
columns_to_keep = ['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'MSRP']
df = df[columns_to_keep]

# Drop rows with missing values
df = df.dropna()

# Rename columns for simplicity
df.columns = ['Year', 'Horsepower', 'Cylinders', 'Mileage', 'Price']

# Calculate Car Age
df['Age'] = 2025 - df['Year']  # Assuming current year is 2025

# Save cleaned dataset
df.to_csv('cleaned_cars.csv', index=False)
print("Preprocessing complete. Dataset saved as 'cleaned_cars.csv'")
