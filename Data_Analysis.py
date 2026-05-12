import pandas as pd
import numpy as np
import csv
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
from sklearn.preprocessing import StandardScaler , LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split,KFold, cross_val_score, RandomizedSearchCV,TimeSeriesSplit
from scipy.stats import randint, uniform 
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo 
from sklearn.linear_model import LogisticRegression


steel_industry_energy_consumption = fetch_ucirepo(id=851) 

X = steel_industry_energy_consumption.data.features 
Y = steel_industry_energy_consumption.data.targets 
df = steel_industry_energy_consumption.data.features.copy()
df['Load_Type'] = steel_industry_energy_consumption.data.targets
df['hour'] = (df['NSM'] // 3600).astype(int)


def analysis_data():

    

    print("Little detail of the like of hte data which are we working on")
    print(f"Columns name and type : {X.columns.tolist()}")
    print(f"column name of the target columns : {Y.columns.tolist()}")
    print(f"Data types of the columns : {X.dtypes}")
    print(f"Other type of detail : {X.describe()}")
    print(f"Length of the total data: {len(X)}")

    # now we will do all of the data analysis using the like of the exploratory data analysis and this is how we done our work efficiently 

    # this is the plot Number One to see the and this is what we used this 
    # Plot No 1
    hr_load = df.groupby(['hour', 'Load_Type'])['Usage_kWh'].mean().reset_index()

    plt.figure(figsize=(12, 5))
    for lt in df['Load_Type'].unique():
        subset = hr_load[hr_load['Load_Type'] == lt]
        plt.plot(subset['hour'], subset['Usage_kWh'], linewidth=2, label=lt)

    plt.title('1. Average Usage (kWh) by Hour & Load Type')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Usage (kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot1_usage_vs_hour.png', dpi=150)

    # like of the this is the second plot in which we learn the lke of hte waste energy change on thel iike of the weekday and vs 
    # the waste energy on the weekend days like of the from mondday to friday and from friday to sunday
    # plot No 2
    weekday = df[df['WeekStatus'] == 'Weekday']
    weekend = df[df['WeekStatus'] == 'Weekend']
    load_types = ['Light_Load', 'Medium_Load', 'Maximum_Load']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Prepare data
    wd_data = [weekday[weekday['Load_Type'] == lt]['Lagging_Current_Reactive.Power_kVarh'].dropna().values for lt in load_types]
    we_data = [weekend[weekend['Load_Type'] == lt]['Lagging_Current_Reactive.Power_kVarh'].dropna().values for lt in load_types]

    # Weekday
    axes[0].boxplot(wd_data, tick_labels=load_types)
    axes[0].set_title('Weekday')
    axes[0].set_ylabel('Lagging Reactive Power (kVarh)')
    axes[0].grid(True, alpha=0.3)

    # Weekend
    axes[1].boxplot(we_data, tick_labels=load_types)
    axes[1].set_title('Weekend')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('2. Lagging Reactive Power: Weekday vs Weekend', fontsize=14)
    plt.tight_layout()
    plt.savefig('plot2_reactive_power_weekstatus.png', dpi=150)

    # Alternative of the Plot 3
    print(df['Leading_Current_Reactive_Power_kVarh'].describe())
    print(df['Leading_Current_Reactive_Power_kVarh'].value_counts().head(10))
    "we will skip it because the like of the this is the value the near of the zero"

    # Plot No 4:
    co2_pivot = df.pivot_table(values='CO2(tCO2)', index='hour', columns='Load_Type', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(12, 6))
    co2_pivot.plot(ax=ax, color=['green', 'orange', 'red'], linewidth=2)

    ax.set_title('4. CO2 Emissions by Hour & Load Type')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('CO2 (tCO2)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot4_co2_vs_hour.png', dpi=150)

    # this is the liek of the plot used for the liek of the different type of the thing 
    #plot No 5
    fig, ax = plt.subplots(figsize=(10, 6))

    df.pivot(columns='Load_Type', values='Lagging_Current_Power_Factor').plot.kde(
        ax=ax, color=['green', 'orange', 'red'], linewidth=2)

    ax.set_title('5. Lagging Power Factor Density by Load Type')
    ax.set_xlabel('Lagging Power Factor')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot5_power_factor_kde.png', dpi=150)

    # plot number 6
    max_load = df[df['Load_Type'] == 'Maximum_Load']

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = max_load.pivot_table(index='Day_of_week', columns='hour', values='Usage_kWh', aggfunc='count')
    pivot = pivot.reindex(day_order)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Record Count'})

    ax.set_title('6. Maximum Load: Count per Day & Hour')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig('plot6_maxload_heatmap.png', dpi=150)

    # plot No 7
    fig, ax = plt.subplots(figsize=(10, 6))

    df.pivot(columns='Load_Type', values='Leading_Current_Power_Factor').plot.kde(
        ax=ax, color=['green', 'orange', 'red'], linewidth=2)

    ax.set_title('7. Leading Power Factor Density by Load Type')
    ax.set_xlabel('Leading Power Factor')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot7_leading_pf_kde.png', dpi=150)
    # plot NO 8
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(range(1000), df['Usage_kWh'].iloc[:1000], color='steelblue', linewidth=1)

    ax.set_title('8. Raw Usage (kWh) - First 1000 Records')
    ax.set_xlabel('Row Index (Time Order)')
    ax.set_ylabel('Usage (kWh)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot8_raw_timeseries.png', dpi=150)
    # plot NO 9
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {'Light_Load': 'green', 'Medium_Load': 'orange', 'Maximum_Load': 'red'}

    df.plot.scatter(
        x='Usage_kWh',
        y='Lagging_Current_Reactive.Power_kVarh',
        c=df['Load_Type'].map(colors),
        ax=ax,
        alpha=0.3,
        s=5
    )

    ax.set_title('9. Usage vs Lagging Reactive Power by Load Type')
    ax.set_xlabel('Usage (kWh)')
    ax.set_ylabel('Lagging Reactive Power (kVarh)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plot9_usage_vs_reactive.png', dpi=150)
