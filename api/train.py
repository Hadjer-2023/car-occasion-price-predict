
import os
import numpy as np 
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

basedir = os.path.abspath(os.path.dirname(__file__))
train_data = os.path.join(basedir, "../data/train.csv")
test_data = os.path.join(basedir, "../data/test.csv")

def load_data():
    df_train = pd.read_csv(train_data)
    df_test = pd.read_csv(test_data)
    return df_train, df_test

def preprocess_data(df_train, df_test):
    features = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Fuel_Type', 'Transmission', 'Owner_Type']
    # Vectoriser la colonne "Name" en utilisant TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_name = vectorizer.fit_transform(df_train['Name'])
    X_test_name = vectorizer.transform(df_test['Name'])
    # Appliquer le clustering k-means
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X_train_name)
    # Ajouter les étiquettes de cluster à la colonne "Name"
    df_train['Name_Cluster'] = kmeans.labels_
    df_test['Name_Cluster'] = kmeans.predict(X_test_name)
    # Nettoyer et préparer les données
    # train
    df_train['Engine'] = df_train['Engine'].str.replace(' CC', '', regex=False)
    df_train['Mileage'] = df_train['Mileage'].str.replace(' kmpl| km/kg', '', regex=True).astype(float)
    df_train['Power'] = df_train['Power'].str.replace(' bhp', '').replace('null', np.nan).astype(float)
    # test
    df_test['Engine'] = df_test['Engine'].str.replace(' CC', '', regex=False)
    df_test['Mileage'] = df_test['Mileage'].str.replace(' kmpl| km/kg', '', regex=True).astype(float)
    df_test['Power'] = df_test['Power'].str.replace(' bhp', '').replace('null', np.nan).astype(float)
    # Supprimer la colonne New_Price
    df_train = df_train.drop(['New_Price'], axis=1)
    df_test = df_test.drop(['New_Price'], axis=1)
    # Séparer les données d'entraînement en features et cible
    X_train = df_train[features + ['Name_Cluster']]
    y_train = df_train['Price']
    # Séparer les données de test en features seulement
    X_test = df_test[features + ['Name_Cluster']]

    return X_train, y_train, X_test

def train_and_evaluate_model(X, y):
    pipeline = create_pipeline()
    # Diviser les données d'entraînement en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # Entraîner le modèle sur les données d'entraînement
    pipeline.fit(X_train, y_train)
    # Évaluer les performances du modèle sur les données de validation
    score = pipeline.score(X_val, y_val)
    print('Score de la pipeline sur les données de validation :', score)
    # Faire des prédictions sur les données de validation
    y_pred = pipeline.predict(X_val)
    # Coefficient de détermination (R²)
    r2 = r2_score(y_val, y_pred)
    print("R² :", r2)
    # Erreur moyenne absolue (MAE)
    mae = mean_absolute_error(y_val, y_pred)
    print("MAE :", mae)
    # Erreur quadratique moyenne (MSE)
    mse = mean_squared_error(y_val, y_pred)
    print("MSE :", mse)
    # Erreur quadratique moyenne racine (RMSE)
    rmse = np.sqrt(mse)
    print("RMSE :", rmse)

    # return pipeline
    return {
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }

def create_pipeline():
    num_cols = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats']
    cat_cols = ['Fuel_Type', 'Transmission', 'Owner_Type', 'Name_Cluster']
    # Définir les transformations pour les colonnes numériques et catégorielles
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Combiner les transformations en une seule étape de prétraitement
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    # Définir le modèle de régression
    model = GradientBoostingRegressor(random_state=42)
    # Créer la pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline

def predict_test_data(pipeline, df_test):


    features = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Fuel_Type', 'Transmission', 'Owner_Type']

    # Transformer les données de test à l'aide du pipeline
    df_test = df_test[features + ['Name_Cluster']]
    X_test = pipeline.named_steps['preprocessor'].transform(df_test)

    # Faire des prédictions sur les données de test
    y_pred = pipeline.named_steps['model'].predict(X_test)

    return y_pred

def gradient_boosting_regressor(hyper_params):
    # Charger les données
    df_train, df_test = load_data()
    # Prétraiter les données
    X, y, df_test = preprocess_data(df_train, df_test)
    # Entraîner et évaluer le modèle
    scoring = train_and_evaluate_model(X, y)
    return scoring 

def train(model, hyper_params):
    print()
    print()
    print(model)
    print(hyper_params)
    print()
    print()
    # return { "back": "ok" }
    return gradient_boosting_regressor('')





# # load dataset 
# iris = load_iris()

# # features && target
# X = iris.data
# y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # train model
# model = LogisticRegression(multi_class="auto", max_iter=100)
# model.fit(X_train, y_train)

# # score
# model.score(X_train, y_train)

# # predictions
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print()
# print(' > Predictions')
# print(y_pred)
# print()
# print(' > Accuracy: ' + str(accuracy))
# print()

# dump(model, 'model.joblib')

# print(' > Model saved successfull')
# print()