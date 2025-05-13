import pandas as pd
import numpy as np
from datetime import datetime


# Load the dataset and clean it
Water_Quality = pd.read_csv("/Users/levilina/Documents/Coding/TidyTuesday/2025/Week_20_Water_Quality_Sydney/water_quality.csv")
Weather = pd.read_csv("/Users/levilina/Documents/Coding/TidyTuesday/2025/Week_20_Water_Quality_Sydney/weather.csv")

#Exploring the data
print("\n--- Exploring the datasets ---")

# Examine water quality data structure
print("\nWater Quality Dataset:")
print(f"shape: {Water_Quality.shape} (rows, columns)")
print("\nFirst 3 rows:")
print(Water_Quality.head(3))
print("\nColumn types:")
print(Water_Quality.dtypes)

# And also examine the weather data structure
print("\nWeather Dataset:")
print(f"shape: {Weather.shape} (rows, columns)")
print("\nFirst 3 rows:")
print(Weather.head(3))
print("\nColumn types:")
print(Weather.dtypes)

# And examine value ranges for key variables
print("\nKey measurements ranges:")
print(f"Enterococci range: {Water_Quality['enterococci_cfu_100ml'].min()} - {Water_Quality['enterococci_cfu_100ml'].max()}")
print(f"Rainfall range: {Weather['precipitation_mm'].min()} - {Weather['precipitation_mm'].max()}")
