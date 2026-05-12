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



    def predict_data():
        df=data_cleaning(X,Y)
        df=extra_feature(df)
        target='Load_Type'


        removed_feature=[target,'Leading_Current_Power_Factor','Leading_Current_Reactive_Power_kVarh']
        numerical_features=df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object','category']).columns.tolist()

        final_numerical_col=[col for col in numerical_features if col not in removed_feature]
        final_categorical_col=[col for col in categorical_features if col not in removed_feature]

        x=df[final_numerical_col+ final_categorical_col]
        y=df[target]
        

        train_size=int(len(df)* 0.8)

        x_train=x.iloc[:train_size]
        x_test=x.iloc[train_size:]
        y_train=y.iloc[:train_size]
        y_test=y.iloc[train_size:]

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)


        preprocessor=ColumnTransformer(
            [
                ('num', StandardScaler(), final_numerical_col),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), final_categorical_col)
            ]
        )

        pipelines=Pipeline(
            [
                ('pre', preprocessor),
                ('model',xgb.XGBClassifier(random_state=42))
            ]
        )


        param_dist = {
        'model__n_estimators': randint(200, 600),
        'model__learning_rate': uniform(0.01, 0.3),
        'model__max_depth': randint(4, 12),
        'model__subsample': uniform(0.6, 0.4),
        'model__colsample_bytree': uniform(0.6, 0.4),
        'model__reg_lambda': uniform(0, 10),
        'model__reg_alpha': uniform(0, 5),
        'model__gamma': uniform(0, 5),
        'model__min_child_weight': randint(1, 10)
    }
        
        search = RandomizedSearchCV(
            estimator=pipelines,
            param_distributions=param_dist,
            n_iter=100,  # Increase if you have time
            cv=TimeSeriesSplit(n_splits=5), # using the cross validation with the 5 cross cv 
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        
        search.fit(x_train,y_train)

        print("Best Parameters:", search.best_params_)
        print("Best CV Accuracy:", search.best_score_)


        y_pred=search.best_estimator_.predict(x_test)

        
        
        accuracy=accuracy_score(y_test, y_pred)


        print(f"accuracy for xg: {accuracy:.3f}")
        print(f"Correct_prediction: {accuracy*100:.1f}")

        print("Classification Report")
        print(classification_report(y_test,y_pred))

        print("\nCONFUSION MATRIX:")
        print(confusion_matrix(y_test, y_pred))



    def unique_value_check():
        numerical_features=df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_features:
            print(f"{col}: {df[col].nunique()} unique values")
            print(df[col].unique()[:10])
            print()



    def extra_feature(df):
        df = df.copy()
        df['hour'] = (df['NSM'] // 3600).astype(int)
        df['Time_of_day'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=False)
        df['Is_Business_Hour'] = df['hour'].between(8, 17).astype

        return df







    if __name__ == "__main__":
        predict_data()




