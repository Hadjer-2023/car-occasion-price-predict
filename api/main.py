# main.py

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
from fastapi import HTTPException

app = FastAPI()

# Load the prediction model from the file model.joblib
model = joblib.load("model.joblib")

templates = Jinja2Templates(directory="C:\\Users\\Utilisateur2\\Documents\\FORM-DEVIA\\4-260224-220324 REGRESSION-CLUSTERING\\5Projet\\car-occasion-price-predict\\web\\templates\\")

class PredictionForm(BaseModel):
    year: int
    kilometers_driven: int
    fuel_type: str
    transmission: str
    owner_type: str
    mileage: float
    engine: int
    power: float
    seats: int

# Example data for testing
example_data = {
    "year": 2014,
    "kilometers_driven": 40929,
    "fuel_type": "CNG",
    "transmission": "Manual",
    "owner_type": "First",
    "mileage": 32.26,
    "engine": 998,
    "power": 58.2,
    "seats": 4
}

@app.post("/predict", response_class=HTMLResponse)
async def predict(form: PredictionForm = Form(...)):
    try:
        # Prepare data for prediction
        features = [form.year, form.kilometers_driven, form.fuel_type, form.transmission, form.owner_type, form.mileage, form.engine, form.power, form.seats]

        # Make prediction
        predicted_price = model.predict([features])[0]

        # Return the result to the prediction.html page
        return templates.TemplateResponse("prediction.html", {"predicted_price": predicted_price})
    except HTTPException as e:
        # Print the error details to the console
        print(f"An error occurred: {e}")
        
        # Return validation details as JSON
        return JSONResponse(content={"detail": e.errors()}, status_code=e.status_code)
    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/prediction", response_class=HTMLResponse)
async def render_prediction_page(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request, "example_data": example_data})