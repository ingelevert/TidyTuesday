import pydytuesday
import pandas as pd
import numpy as np
from datetime import datetime


# Load the dataset and clean it
Water_Quality = pd.read_csv("/Users/levilina/Documents/Coding/TidyTuesday/2025/Week_20_Water_Quality_Sydney/water_quality.csv")
Weather = pd.read_csv("/Users/levilina/Documents/Coding/TidyTuesday/2025/Week_20_Water_Quality_Sydney/weather.csv")

# 1. Clean the data for water quality
def clean_water_quality(df):
    # Convert date and time columns to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time

    # Create year, month and season columns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['season'] = pd.cut(
        df['month'],
        bins=[0, 3, 6, 9, 12],
        labels=['Summer', 'Autumn', 'Winter', 'Spring'],
    )

    # Handle missing values
    print(f"Missing values before cleaning;\n{df.isna().sum()}")

    # Convert NA values to NaN
    df = df.replace('NA', np.nan)

    # Convert numeric columns to appropriate data types
    numeric_cols = ['enterococci_cfu_100ml', 'water_temperature_c', 'conductivity_ms_cm']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Missing values after cleaning\n{df.isna().sum()}")

    # Create water quality assessment column based on enterococci count
    # Using NSW beachwatch Guidelines, which are:
    # - Good: 0-40 CFU/100ml
    # - Fair: 41-200 CFU/100ml
    # - Poor: >200 CFU/100ml
    bins = [-1, 40, 200, float('inf')]
    labels = ['Good', 'Fair', 'Poor']
    df['quality_category'] = pd.cut(df['enterococci_cfu_100ml'], bins=bins, labels=labels)
    return df


# 2. Data cleaning for the weather data
def clean_weather(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df

# Clean the dataset
water_quality_clean = clean_water_quality(Water_Quality)
weather_clean = clean_weather(Weather)

# Merge datasets on date for analysis that requires both
merged_data = pd.merge(
    water_quality_clean, 
    weather_clean[['date', 'precipitation_mm', 'max_temp_C', 'min_temp_C']], 
    on='date', 
    how='left'
)