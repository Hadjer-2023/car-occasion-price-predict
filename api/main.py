
from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris

# laod dataset 
iris = load_iris()

# load model
model = load('model.joblib')

app = FastAPI()

# define class to make request
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

# define route: http://127.0.0.1:8000/predict
@app.post("/predict") 

# define prédiction
def predict(data : request_body):
    to_predict = [[ data.sepal_length, data.sepal_width, data.petal_length, data.petal_width ]]
    prediction = model.predict(to_predict)[0]
    return {'class' : iris.target_names[prediction]}
