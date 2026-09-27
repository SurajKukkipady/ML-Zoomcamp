import pandas as pd
import numpy as np

df = pd.read_csv("car_fuel_efficiency_2026.csv")

print(len(df))
print(df.isnull().sum())
print("Max fuel efficiency:", df["fuel_efficiency_mpg"].max())

median_before = df["horsepower"].median()
mode_horsepower = df["horsepower"].mode().iloc[0]

print("Median horsepower before filling:", median_before)
print("Most frequent horsepower:", mode_horsepower)

filled_horsepower = df["horsepower"].fillna(mode_horsepower)
print("Median horsepower after filling:", filled_horsepower.median())

# Matrix operations on cars from Asia
X_df = df[df["origin"] == "Asia"][['vehicle_weight', 'model_year']].head(7)
X = X_df.to_numpy()
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y

print("X:\n", X)
print("XTX:\n", XTX)
print("XTX inverse:\n", XTX_inv)
print("w:", w)
print("Sum of all elements of w:", w.sum())