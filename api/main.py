
import os
import pandas as pd
import joblib

from fastapi import Depends, FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from livereload import Server

from pydantic import BaseModel, Field
from joselib import jwt
from passlib.context import CryptContext

from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional, Dict, Any

from train import train 


""" 
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    > SETUP APP && MAIN VARIABLES
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
""" 

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# openssl rand -hex 32
SECRET_KEY = "ae16b4ed21dfe4905a618cf3647851d5a997cf3f8e27110b0cb33b4e70aadbc5"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 10

basedir = os.path.abspath(os.path.dirname(__file__))

MODELS = ["lr", "l", "r", "en", "abr", "gbr", "rfr"]
PROCESSES = ["cv", "gs"]


""" 
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    > FAKE DB 
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
""" 

hashed_password = pwd_context.hash("ok")

print(hashed_password)

allowed = [
    { "name": "hadjer", "password": hashed_password },
    { "name": "patrick", "password": hashed_password },
    { "name": "raph", "password": hashed_password },
    { "name": "salah", "password": hashed_password }
]


""" 
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    > DEFINE SCHEMAS 
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
""" 

class Dev(BaseModel):
    name: str = Field(default=None)
    password: str = Field(default=None)


class Predict(BaseModel):
    brand: str = Field(default=None)
    location: str = Field(default=None)
    year: str = Field(default=None)
    kilometers: str = Field(default=None)
    fuel: str = Field(default=None)
    transmission: str = Field(default=None)
    owner_type: str = Field(default=None)
    engine: str = Field(default=None)
    power: str = Field(default=None)
    seats: str = Field(default=None)
    mileage: str = Field(default=None)


class Model(BaseModel):
    model: str = Field(default=None)
    hyper_params: Dict[str, Any] 


""" 
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    > HOME && AUTH  
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
""" 

@app.get("/")
def dev(request: Request):
    context = { "request": request, "message": "Dev Zone" }
    return templates.TemplateResponse("index.html", context)


@app.post("/auth")
async def login(dev: Dev, request: Request):
    data = await request.json()
    print(data)
    if "name" not in data or "password" not in data:
        return { "success" : False, "message": "Nom ou mot de passe incorrect" }
    dev = authenticate_dev(data["name"], data["password"])
    if not dev:
        return { "success" : False, "message": "Nom ou mot de passe incorrect" }

    access_token = create_access_token(dev)

    return { "success" : True, "access_token": access_token }


""" 
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    > PROTECTED
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
""" 

async def get_current_dev(access_token):
    if not access_token:
        raise HTTPException(status_code=401, detail="Non authentifié")
    dev = get_dev_from_db(access_token)
    if dev is None:
        raise HTTPException(status_code=401, detail="Non authentifié")

    return dev


@app.get("/protected/{access_token}")
async def protected(request: Request, access_token: str, dev: dict = Depends(get_current_dev)):
    print(dev)
    context = { "request": request, "dev": dev['name'] }
    return templates.TemplateResponse("protected.html", context)


@app.post("/train")
async def train_protected(request: Request):
    data = await request.json()
    print(data)
    if 'access_token' not in data or data['access_token'] is None:
        return { "fail": "Token expiré" }
    dev = get_dev_from_db(data['access_token'])
    if dev is None:
        return { "fail": "not_allowed" }
    model = data["model"]
    process = data["process"]
    hyparams = data["hyparams"]
    if model in MODELS and process in PROCESSES:
        return train(model, process, hyparams)
    else: 
        return { "fail": True, "message": "Model ou process incorrect" }


""" 
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    > PREDICT FROM APPLICATION WEB localhost:5000
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
""" 

@app.options("/predict", include_in_schema=False)
def options_predict():
    return JSONResponse({ "message": "Options request allowed" }, status_code=200)


@app.post("/predict")
def predict(data: Predict):
    data = check_data(data)
    print(data)
    to_predict = pd.DataFrame({

        "Name": [data.brand],
        "Location": [data.location],
        "Year": [data.year],
        "Kilometers_Driven": [data.kilometers],
        "Fuel_Type": [data.fuel],
        "Transmission": [data.transmission],
        "Owner_Type": [data.owner_type],
        "Mileage": [data.mileage],
        "Engine": [data.engine],
        "Power": [data.power],
        "Seats": [data.seats]

    })

    # RandomForestRegressor_cross_val RandomForestRegressor_grid_search
    # GradientBoostingRegressor_cross_val GradientBoostingRegressor_grid_search
    the_model = "RandomForestRegressor_grid_search"
    loaded_model = joblib.load('./tmp/' + the_model + '.joblib')
    prediction = loaded_model.predict(to_predict)

    return JSONResponse({ "prediction": prediction[0] }, status_code=200)
 

""" 
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    > FUNCTIONS 
    ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
""" 

def authenticate_dev(name: str, password: str):
    dev = next((d for d in allowed if d["name"] == name), None)
    if not dev:
        return None
    if not pwd_context.verify(password, dev["password"]):
        return None

    return dev


def create_access_token(dev: dict):
    payload = {
        "sub": dev["name"],
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    access_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return access_token


def get_dev_from_db(access_token: str):
    for dev in allowed:
        if access_token_decode(access_token).get("sub") == dev["name"]:
            return dev
    return None


def access_token_decode(access_token: str):
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Jeton d'accès expiré")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Jeton d'accès invalide")


def check_data(data):
    data.year = None if len(data.year) == 0 else float(data.year)
    data.kilometers = None if len(data.kilometers) == 0 else float(data.kilometers)
    data.engine = None if len(data.engine) == 0 else float(data.engine)
    data.power = None if len(data.power) == 0 else float(data.power)
    data.seats = None if len(data.seats) == 0 else float(data.seats)
    data.mileage = None
    
    return data
