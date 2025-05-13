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


def analyze_rainfall_effect_improved(df):
    # Use the same data preparation as before
    data = df.copy()
    data['log_enterococci'] = np.log1p(data['enterococci_cfu_100ml'])
    data = data.dropna(subset=['log_enterococci', 'precipitation_mm'])
    
    # Group rainfall into categories
    bins = [-1, 0, 5, 15, float('inf')]
    labels = ['None (0 mm)', 'Light (0-5 mm)', 'Moderate (5-15 mm)', 'Heavy (>15 mm)']
    data['rain_category'] = pd.cut(data['precipitation_mm'], bins=bins, labels=labels)
    
    # Calculate sample sizes and median values for each category
    category_stats = data.groupby('rain_category').agg(
        sample_size=('enterococci_cfu_100ml', 'count'),
        median=('enterococci_cfu_100ml', 'median')
    )
    
    # Enhanced boxplot with improvements
    plt.figure(figsize=(14, 8))
    
    # Create boxplot with color-blind friendly palette
    ax = sns.boxplot(
        data=data,
        x='rain_category', 
        y='enterococci_cfu_100ml',
        palette=["#5154a4", "#56a8cb", "#5dc863", "#cae931"],  # Color-blind friendly blue-green gradient
        width=0.6,
        showfliers=True,  # Show outliers
        flierprops={'marker':'o', 'markersize':3, 'alpha':0.5}  # Smaller, semi-transparent outliers
    )
    
    # Add sample size to each category
    for i, (category, stats) in enumerate(category_stats.iterrows()):
        ax.text(i, data['enterococci_cfu_100ml'].min() / 2, 
                f"n={stats['sample_size']}", 
                ha='center', color='black', fontsize=10)
    
    # Connect median points with a line to emphasize trend
    median_line_data = []
    for i, (category, stats) in enumerate(category_stats.iterrows()):
        median_line_data.append((i, stats['median']))
    
    xs, ys = zip(*median_line_data)
    plt.plot(xs, ys, 'r--', linewidth=2, alpha=0.7, label='Median trend')
    
    # Add arrow showing the increase
    first_median = category_stats.iloc[0]['median']
    last_median = category_stats.iloc[-1]['median']
    increase_factor = last_median / first_median
    
    plt.annotate(
        f"~{increase_factor:.1f}× increase in median\nwith heavy rainfall", 
        xy=(3, last_median), 
        xytext=(2, last_median * 3),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Add water quality thresholds
    plt.axhline(y=40, color='green', linestyle='-', alpha=0.5, label='Good threshold (40 CFU)')
    plt.axhline(y=200, color='red', linestyle='-', alpha=0.5, label='Poor threshold (200 CFU)')
    plt.fill_between([-0.5, 3.5], 0, 40, color='green', alpha=0.1)
    plt.fill_between([-0.5, 3.5], 40, 200, color='yellow', alpha=0.1)
    plt.fill_between([-0.5, 3.5], 200, data['enterococci_cfu_100ml'].max(), color='red', alpha=0.1)
    
    # Improve y-axis tick labels
    plt.yscale('log')
    plt.yticks([1, 10, 40, 100, 200, 1000, 10000, 100000], 
              ['1', '10', '40\n(Good)', '100', '200\n(Poor)', '1,000', '10,000', '100,000'])
    
    # Add labels and title
    plt.title('Impact of Rainfall on Beach Water Quality', fontsize=20)
    plt.xlabel('Rainfall Category', fontsize=16)
    plt.ylabel('Enterococci (CFU/100ml) - Log Scale', fontsize=16)
    
    # Adjust grid to be lighter
    plt.grid(True, alpha=0.2)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig("2025/Week_20_Water_Quality_Sydney/Graphs/enterococci_by_rain_improved.png", dpi=300)
    plt.close()
    
    print("Enhanced rainfall effect analysis completed.")
    return data

# Call the improved function
analyze_rainfall_effect_improved(merged_data)

