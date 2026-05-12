import pandas as pd
import numpy as np
import csv
import re

steel_industry_energy_consumption = fetch_ucirepo(id=851) 

X = steel_industry_energy_consumption.data.features 
Y = steel_industry_energy_consumption.data.targets 
df = steel_industry_energy_consumption.data.features.copy()
df['Load_Type'] = steel_industry_energy_consumption.data.targets
df['hour'] = (df['NSM'] // 3600).astype(int)


def data_cleaning(X,Y):
    df = X.copy()
    df['Load_Type'] = Y['Load_Type'].values

    print("Missing value check ===========================================")

    print("To see if how many of the values are missing ")
    print(f"Missing Values: {X.isnull().sum()}")
    print("THis data has the Zero missing valus and this data set is prone to Missng values")

    print("Duplicate Value check ==========================================")

    print("to check the duplicate values")
    print(f"Duplicate Values Check : {X.duplicated().sum()}")
    "this data set contain some of the missing values so we are dropping all of those missing values from this datset"
    df=df.drop_duplicates()

    print("Adding Data Quality Checks ======================================")

    df = df[df['NSM'] >= 0]
    df = df[df['NSM'] <= 86400]
    df = df[df['Usage_kWh'] >= 0]
    df = df[df['Lagging_Current_Reactive.Power_kVarh'] >= 0]
    df = df[df['Leading_Current_Reactive_Power_kVarh'] >= 0]
    df = df[df['CO2(tCO2)'] >= 0]
    df = df[df['Lagging_Current_Power_Factor'] >= 0]
    df = df[df['Lagging_Current_Power_Factor'] <= 100]
    df = df[df['Leading_Current_Power_Factor'] >= 0]
    df = df[df['Leading_Current_Power_Factor'] <= 100]
    df = df[df['WeekStatus'].isin(['Weekday', 'Weekend'])]
    df = df[df['Day_of_week'].isin(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])]
    df = df[df['Load_Type'].isin(['Light_Load', 'Medium_Load', 'Maximum_Load'])]

    df = df.reset_index(drop=True)

    print("Adding outlier detection ========================================")

    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(numerical_features):
        axes[idx].boxplot(df[col])
        axes[idx].set_title(f'{col} - Outlier Check')
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        print(f"{col}: {len(outliers)} outliers detected")

    # Hide unused subplots
    for idx in range(len(numerical_features), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig("Outlier_checker.png")

    return df
