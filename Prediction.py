
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

