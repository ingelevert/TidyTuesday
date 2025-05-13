import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D


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


# Data cleaning --------------------------------------------------------------------------------------------------------------------

# Clean water quality data
def clean_water_quality(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.replace('NA', np.nan)
    numeric_cols = ['enterococci_cfu_100ml', 'water_temperature_c', 'conductivity_ms_cm']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # water quality catagories, good ≤ 40 CFU/100ml, fair 41-200 CFU/100ml, poor: >200 CFU/100ml
    bins = [-1, 40, 200, float('inf')]
    labels = ['Good', 'Fair', 'Poor']
    df['quality_category'] = pd.cut(df['enterococci_cfu_100ml'], bins=bins, labels=labels)
    return df

# Clean weather data
def clean_weather(df):
    df['date'] = pd.to_datetime(df['date'])
    return df

# Apply
Water_Quality_clean = clean_water_quality(Water_Quality)
Weather_clean = clean_weather(Weather)

# Merge datasets
merged_data = pd.merge(
    Water_Quality_clean,
    Weather_clean[['date', 'precipitation_mm', 'max_temp_C']],
    on='date',
    how='left'
)

# Analyze and visualize -------------------------------------------------------------------------------------------------

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# 1. Distribution of enterococci levels (log)
plt.figure(figsize=(10, 6))
plt.hist(np.log1p(Water_Quality_clean['enterococci_cfu_100ml'].dropna()), bins=30)
plt.title('Distribution of Enterococci Levels (log)')
plt.xlabel('log(enterococci CFU/100ml + 1)')
plt.ylabel('Frequency')
plt.savefig('/Users/levilina/Documents/Coding/TidyTuesday/2025/Week_20_Water_Quality_Sydney/enterococci_distribution.png')
plt.close()


# 2. Scatterplot enterococci vs. rainfall
plt.figure(figsize=(12, 8))

# Create scatter plot iwht points colored by the quality category made earlier
scatter = plt.scatter(
    merged_data['precipitation_mm'],
    merged_data['enterococci_cfu_100ml'],
    c=merged_data['quality_category'].map({'Good': 0, 'Fair': 1, 'Poor': 2}),
    cmap='RdYlGn_r', #RYG colormap (reversed)
    alpha=0.5,
    s=30, # size of the points
    edgecolor='none'
)

# Add log sclae for y-axis
plt.yscale('log')

# Reference lines for water quality thresholds
plt.axhline(y=40, color='green', linestyle='--', alpha=0.7, label='Good threshold (40 CFU)')
plt.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='Poor threshold (200 CFU)')

# Custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Good'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Fair'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Poor')
]
plt.legend(handles=legend_elements, title='Water Quality Category')

# Add a trend line with numpy polyfit
valid_data = merged_data.dropna(subset=['precipitation_mm', 'enterococci_cfu_100ml'])
x = valid_data['precipitation_mm']
y = np.log10(valid_data['enterococci_cfu_100ml'] + 1)

if len(valid_data) > 10:
    coeffs = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coeffs)
    x_trend = np.linspace(0, merged_data['precipitation_mm'].max(), 100)
    y_trend = 10 ** polynomial(x_trend) - 1

    # Plot the trend line
    plt.plot(x_trend, y_trend, color='blue', linestyle='-', lw=2,
             label=f'Trend (y) ∝ 10^{coeffs[0]:.4f}x)')
    plt.legend()

# add text showing correlation value
if len(valid_data) > 10:
    correlation = merged_data['precipitation_mm'].corr(np.log10(merged_data['enterococci_cfu_100ml'] + 1))
    plt.text(0.05, 0.95, f"Correlation: {correlation:.3f}", 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()

plt.savefig('/Users/levilina/Documents/Coding/TidyTuesday/2025/Week_20_Water_Quality_Sydney/enterococci_vs_rainfall.png')
plt.close()

print("Rainfall vs enterococci scatter plot created successfully.")