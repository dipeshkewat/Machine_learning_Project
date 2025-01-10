import pandas as pd

# Load the dataset
df = pd.read_csv('cars.csv')

# Select relevant columns
columns_to_keep = ['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'MSRP']
df = df[columns_to_keep]

# Drop rows with missing values
df = df.dropna()

# Rename columns for simplicity
df.columns = ['Year', 'Horsepower', 'Cylinders', 'HighwayMPG', 'Price']

# Remove outliers (e.g., very high prices)
df = df[df['Price'] < 100000]

# Save the cleaned dataset
df.to_csv('cleaned_cars.csv', index=False)
print("Dataset cleaned and saved to 'cleaned_cars.csv'")
