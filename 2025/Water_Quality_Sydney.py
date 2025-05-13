import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from datetime import datetime, timedelta
import os

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

# ---- Data analysis --------------------------------------------------------------------------------------------------------------------

# Firstly, lets make a nice matplotlib default style
# Set up a nicer matplotlib default for LinkedIn-worthy charts
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 100
})

print("Analyzing the data...")

# Lets do the analysis by Region: Water Quality Categories

def analyze_quality_by_region(df):
    # Count quality categories by region
    quality_by_region = pd.crosstab(
        df['region'], 
        df['quality_category'],
        normalize='index'
    ) * 100
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    quality_by_region.plot(
        kind='bar', 
        stacked=True,
        ax=ax,
        color=['#1b9e77', '#d95f02', '#7570b3'],  # Colorblind-friendly palette
        width=0.7
    )
    for i, region in enumerate(quality_by_region.index):
        cumulative = 0
        for quality, percentage in quality_by_region.loc[region].items():
            if percentage > 5:  # Only label if sufficient space
                ax.text(
                    i, 
                    cumulative + percentage/2, 
                    f"{percentage:.0f}%", 
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color='white' if quality != 'Good' else 'black'
                )
            cumulative += percentage
    
    ax.set_title('Water Quality by Region in Sydney')
    ax.set_xlabel('Region')
    ax.set_ylabel('Percentage')
    ax.set_ylim(0, 100)
    ax.legend(title='Water Quality', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("/Users/levilina/Documents/Coding/TidyTuesday/2025/Week_20_Water_Quality_Sydney/Graphs/quality_by_region.png", dpi=300)
    plt.close()

    return quality_by_region

# Call the water quality by region analysis function
quality_results = analyze_quality_by_region(Water_Quality_clean)
print("Water quality analysis by region completed.")

#------------------------------------------------------------------------------------------

#Rainfall vs Bacteria (Enterococci)

def analyze_rainfall_effect(df):
    data = df.copy()
    data['log_enterococci'] = np.log1p(data['enterococci_cfu_100ml'])
    data = data.dropna(subset=['log_enterococci', 'precipitation_mm'])

    try:
        m, b = np.polyfit(data['precipitation_mm'], data['log_enterococci'], 1)
        fit_valid = True
    except:
        fit_valid = False
        m, b = 0, 0

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=data,
        x='precipitation_mm',
        y='log_enterococci',
        hue='region',
        alpha=0.6,
        ax=ax
    )

# We will add a regression line if the fit is valid
    if fit_valid:
        x_range = np.linspace(0, data['precipitation_mm'].max(), 100)
        y_pred = m * x_range + b
        ax.plot(x_range, y_pred, 'r-', linewidth=2, 
                label=f'y = {m:.3f}x + {b:.2f}')
        # Calculate correlation coefficient
        corr = np.corrcoef(data['precipitation_mm'], data['log_enterococci'])[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_title('Rainfall vs. Enterococci Levels')
    ax.set_xlabel('Precipitation (mm)')
    ax.set_ylabel('Log(Enterococci CFU/100ml + 1)')
    
    # Move legend outside plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("2025/Week_20_Water_Quality_Sydney/Graphs/enterococci_vs_rainfall.png", dpi=300)
    plt.close()
    
    # Analyze rainfall effect with categories
    # Group rainfall into categories
    bins = [-1, 0, 5, 15, float('inf')]
    labels = ['None', 'Light (0-5mm)', 'Moderate (5-15mm)', 'Heavy (>15mm)']
    data['rain_category'] = pd.cut(data['precipitation_mm'], bins=bins, labels=labels)
    
    # Boxplot of enterococci by rain category
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=data,
        x='rain_category', 
        y='enterococci_cfu_100ml',
        palette='viridis'
    )
    plt.yscale('log')
    plt.title('Enterococci Levels by Rainfall Category')
    plt.xlabel('Rainfall Category')
    plt.ylabel('Enterococci (CFU/100ml) - Log Scale')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/levilina/Documents/Coding/TidyTuesday/2025/Week_20_Water_Quality_Sydney/Graphs/enterococci_by_rain_category.png", dpi=300)
    plt.close()
    print("Rainfall effect analysis completed.")
    return data

# Call the rainfall analysis function
analyze_rainfall_effect(merged_data)
print("Rainfall effect analysis completed.")
