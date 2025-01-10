import pandas as pd

df = pd.read_csv('cars.csv')

columns_to_keep = ['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'MSRP']
df = df[columns_to_keep]

df = df.dropna()

df.columns = ['Year', 'Horsepower', 'Cylinders', 'Mileage', 'Price']

df['Age'] = 2025 - df['Year']  

df.to_csv('cleaned_cars.csv', index=False)
print("Preprocessing complete. Dataset saved as 'cleaned_cars.csv'")
