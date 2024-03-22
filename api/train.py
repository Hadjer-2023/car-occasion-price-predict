
import os
import time
import json
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#   > MAIN VARIABLES
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

base_dir = os.path.abspath(os.path.dirname(__file__))
csv_data = os.path.join(base_dir, "../data/train.csv")

df_original = pd.read_csv(csv_data)

DECIMALS = 4

MODELS = {
    "lr": { "class_": LinearRegression, "name": "LinearRegression" },
    "l": { "class_": Lasso, "name": "Lasso" },
    "r": { "class_": Ridge, "name": "Ridge" },
    "en": { "class_": ElasticNet, "name": "ElasticNet" },
    "abr": { "class_": AdaBoostRegressor, "name": "AdaBoostRegressor" },
    "gbr": { "class_": GradientBoostingRegressor, "name": "GradientBoostingRegressor" },
    "rfr": { "class_": RandomForestRegressor, "name": "RandomForestRegressor" }
}

HYPER_PARAMETERS = {
    "lr": {},
    "l": { "alpha": [0.1, 1.0, 10.0] },
    "r": { "alpha": [0.1, 1.0, 10.0] },
    "en": { "alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9] },
    "abr": { "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0] },
    "gbr": { "n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0] },
    "rfr": { "n_estimators": [10, 50, 100], "max_depth": [3, 5, 10] }
}


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#   > CLEANING
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def clean_df(print_diff=True):
    df_cleaned = df_original.copy()
    df_cleaned = df_cleaned.drop('New_Price', axis=1)
    df_cleaned['Name'] = df_cleaned['Name'].str.split().str[0]
    df_cleaned['Owner_Type'] = df_cleaned['Owner_Type'].str.split().str[0]
    df_cleaned['Year'] = df_cleaned['Year'].astype(float)
    df_cleaned['Kilometers_Driven'] = df_cleaned['Kilometers_Driven'].astype(float)
    df_cleaned['Mileage'] = df_cleaned['Mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
    df_cleaned['Engine'] = df_cleaned['Engine'].str.extract(r'(\d+\.?\d*)').astype(float)
    df_cleaned['Power'] = df_cleaned['Power'].str.extract(r'(\d+\.?\d*)').astype(float)
    df_cleaned.replace('null', pd.NA, inplace=True)
    df_cleaned = df_cleaned.dropna()
    if print_diff:
        display_diff(df_original, df_cleaned)

    return df_cleaned


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#   > PREPROCESSING
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def preprocess(args={}):
    args = get_args({ 

        "print_args": True,
        "title": "preprocess",
        "print_preprocessor": False,
        # NUMERIC
        "imputer_num": False,
        "imputer_num_strategy": "median",
        # CATEGORICAL
        "imputer_cat": False,
        "imputer_cat_strategy": "constant",
        "imputer_cat_fill": "missing",
        "one_hot_handle": "ignore"

    }, args)

    transformers = []

    # ---- ---- NUMERIC
    if "num" in args:
        steps_num = [('scaler', StandardScaler())]
        if args["imputer_num"]:
            steps_num.insert(0, ('imputer', SimpleImputer(strategy=args["imputer_num_strategy"])))
        numeric_pipeline = Pipeline(steps=steps_num)
        transformers.append(('num', numeric_pipeline, args["num"]))

    # ---- ---- CATEGORICAL
    if "cat" in args:
        steps_cat = [('onehot', OneHotEncoder(handle_unknown=args["one_hot_handle"]))]
        if args["imputer_cat"]:
            steps_cat.insert(0, ('imputer', SimpleImputer(strategy=args["imputer_cat_strategy"], fill_value=args["imputer_cat_fill"])))
        categorical_pipeline = Pipeline(steps=steps_cat)
        transformers.append(('cat', categorical_pipeline, args["cat"]))

    preprocessor = ColumnTransformer(transformers=transformers)

    display_preprocessor(preprocessor) if args["print_preprocessor"] else None

    return preprocessor


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#   > TRAINING && SCORING
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def train(model, process, hyparams, feature_selection=False):
    try:
        args = get_args({ 

            "print_args": True,
            "title": "Training " + model + " with process " + process

        }, { "model": model, "process": process, "hyparams": hyparams, "feature_selection": feature_selection })

        # clean
        df_cleaned = clean_df()
        # split
        train_test = get_train_test(df_cleaned)
        # preprocess
        preprocessor = preprocess({
            "num": train_test["X"].select_dtypes(include=['float64']).columns, 
            "cat": train_test["X"].select_dtypes(include=['object']).columns,
            "imputer_num": True, 
            "imputer_cat": True
        })

        print(MODELS[model])

        # train_model = (MODELS[model][class_](), HYPER_PARAMETERS[model]) if process == "gs" else (MODELS[model][class_](), {})
        train_model = (MODELS[model]["class_"](), hyparams) if process == "gs" else (MODELS[model]["class_"](), {})

        X, y, X_train, X_test, y_train, y_test = train_test["X"], train_test["y"], train_test["X_train"], train_test["X_test"], train_test["y_train"], train_test["y_test"]
        
        steps = [('preprocessor', preprocessor), ('model', train_model[0])]
        
        if feature_selection:
            steps.insert(1, ('feature_selection', RFECV(estimator=train_model[0], scoring='r2')))

        pipeline = Pipeline(steps)

        scoring = {}
        name = MODELS[model]["name"]
        start_time = time.time()
        if process == "gs":
            scoring = get_grid_scores(name, train_model[1], pipeline, X_train, y_train, X_test, y_test)
        else:
            scoring = get_cross_scores(name, pipeline, X_train, y_train, X_test, y_test)

        end_time = time.time()

        scoring["execution_time"] = end_time - start_time

        return scoring

    except Exception as e:
        print()
        print()
        print(" > Fail: ")
        print(e)
        print()
        print()
        return { "fail": True, "message": "Une erreur est survenue durant l'entaînement" }


def get_grid_scores(name, hyper_params, pipeline, X_train, y_train, X_test, y_test):
    hyper_params = {f'model__{param_name}': param_values for param_name, param_values in hyper_params.items()}
    grid_search = GridSearchCV(pipeline, hyper_params, cv=5, scoring='r2')
    
    grid_search.fit(X_train, y_train)

    joblib.dump(grid_search.best_estimator_, "./tmp/" + name + "_grid_search.joblib")

    y_pred = grid_search.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2_grid = grid_search.score(X_test, y_test)
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    _best_params = { key.replace('model__', ''): value for key, value in best_params.items() }

    display_scoring({ 
        "model": name, "mae": mae, "mse": mse, "rmse": rmse, "r2_grid": r2_grid, 
        "best_score": best_score, "best_params": _best_params 
    })

    return { "mae": mae, "mse": mse, "rmse": rmse, "r2_grid": r2_grid, "best_score": best_score, "best_params": _best_params }


def get_cross_scores(name, pipeline, X_train, y_train, X_test, y_test):
    r2_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    r2_cross_mean = r2_scores.mean()

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "./tmp/" + name + "_cross_val.joblib")

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2_cross_pred = r2_score(y_test, y_pred)

    display_scoring({ "model": name, "mae": mae, "mse": mse, "rmse": rmse, "r2_cross_pred": r2_cross_pred, "r2_cross_mean": r2_cross_mean })

    return { "mae": mae, "mse": mse, "rmse": rmse, "r2_cross_mean": r2_cross_mean, "r2_cross_pred": r2_cross_pred }


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#   > UTILS
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def get_args(params, args):
    if args is None:
        args = params
    else:
        params.update(args)
        args = params

    display_args(args) if args["print_args"] else None

    return args


def get_train_test(df, args={}):
    args = get_args({ 

        "print_args": True,
        "title": "get_train_test",
        "to_drop": ["Price"],
        "target": "Price",
        "test_size": 0.2,
        "random_state": 0

    }, args)

    X = df.drop(columns=args["to_drop"])
    y = df[args["target"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args["test_size"], random_state=args["random_state"])

    return { "X": X, "y": y, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test }


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#   > DISPLAY
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def display_diff(df, df_cleaned):
    print()
    print('    > DF SHAPE: ', df.shape)
    print()
    print('    > DF CLEANED SHAPE: ', df_cleaned.shape)
    print()


def display_args(args):
    title = args["title"]
    print('')
    print(' > Args for ' + title + ':')
    del args["title"]
    del args["print_args"]
    for key, value in args.items():
        print(f"    {key}: {value}")
    print('')
    print(' --- ' + title + ' in progress...')
    print('')


def display_preprocessor(preprocessor):
    print('')
    print('Preprocessor: ')
    print('')
    print(preprocessor)
    print('')


def display_scoring(data):
    print()
    print(' > Scoring for ' + data["model"] + ':')
    formatted = ""
    del data["model"]
    for metric, value in data.items():
        display = ""
        if isinstance(value, (int, float, np.float64)):
            display = "{:.{}f}".format(value, DECIMALS)
        else:
            display = str(value)
        print("    " + metric +  ": " + display)
    print()