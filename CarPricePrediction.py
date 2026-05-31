from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "cardekho_imputated.csv"

    df = pd.read_csv(data_path, index_col=[0])

    print("Initial data preview:\n")
    print(df.head())
    print("\nData info:\n")
    print(df.info())
    print("\nMissing values:\n")
    print(df.isnull().sum())

    df.drop("car_name", axis=1, inplace=True)
    df.drop("brand", axis=1, inplace=True)

    print("\nData after dropping car_name and brand:\n")
    print(df.head())

    print("\nUnique models:\n")
    print(df["model"].unique())

    num_features = [feature for feature in df.columns if df[feature].dtype != "O"]
    print("\nNum of Numerical Features:", len(num_features))
    cat_features = [feature for feature in df.columns if df[feature].dtype == "O"]
    print("Num of Categorical Features:", len(cat_features))
    discrete_features = [feature for feature in num_features if len(df[feature].unique()) <= 25]
    print("Num of Discrete Features:", len(discrete_features))
    continuous_features = [feature for feature in num_features if feature not in discrete_features]
    print("Num of Continuous Features:", len(continuous_features))

    X = df.drop(["selling_price"], axis=1)
    y = df["selling_price"]

    print("\nFeature sample:\n")
    print(X.head())
    print("\nNumber of unique model values:", len(df["model"].unique()))
    print("\nModel value counts:\n")
    print(df["model"].value_counts())

    le = LabelEncoder()
    X["model"] = le.fit_transform(X["model"])

    print("\nEncoded feature sample:\n")
    print(X.head())

    print(
        "\nUnique counts for seller_type, fuel_type, transmission_type:",
        len(df["seller_type"].unique()),
        len(df["fuel_type"].unique()),
        len(df["transmission_type"].unique()),
    )

    num_features = X.select_dtypes(exclude="object").columns
    onehot_columns = ["seller_type", "fuel_type", "transmission_type"]

    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder(drop="first")

    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, onehot_columns),
            ("StandardScaler", numeric_transformer, num_features),
        ],
        remainder="passthrough",
    )

    X = preprocessor.fit_transform(X)
    print("\nTransformed feature matrix shape:", X.shape)
    print("\nTransformed data preview:\n")
    print(pd.DataFrame(X).head())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\nTrain/Test shapes:", X_train.shape, X_test.shape)

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
    }

    print("\nBeginning model training:\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(
            y_train, y_train_pred
        )
        model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(
            y_test, y_test_pred
        )

        print(name)
        print("Model performance for Training set")
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))
        print("----------------------------------")
        print("Model performance for Test set")
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
        print("=" * 35)
        print()

    knn_params = {"n_neighbors": [2, 3, 10, 20, 40, 50]}
    rf_params = {
        "max_depth": [5, 8, 15, None, 10],
        "max_features": [5, 7, "sqrt", 8],
        "min_samples_split": [2, 8, 15],
        "n_estimators": [50, 100, 150],
    }

    randomcv_models = [
        ("KNN", KNeighborsRegressor(), knn_params),
        ("RF", RandomForestRegressor(random_state=42), rf_params),
    ]

    print("\nHyperparameter tuning:\n")
    model_param = {}
    for name, model, params in randomcv_models:
        random = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=6,
            cv=2,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )
        random.fit(X_train, y_train)
        model_param[name] = random.best_params_

    for model_name in model_param:
        print(f"---------------- Best Params for {model_name} -------------------")
        print(model_param[model_name])

    print("\nRetraining the models with best parameters:\n")
    models = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            min_samples_split=2,
            max_features="sqrt",
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        ),
        "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(
            y_train, y_train_pred
        )
        model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(
            y_test, y_test_pred
        )

        print(name)
        print("Model performance for Training set")
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))
        print("----------------------------------")
        print("Model performance for Test set")
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
        print("=" * 35)
        print()


if __name__ == "__main__":
    main()
