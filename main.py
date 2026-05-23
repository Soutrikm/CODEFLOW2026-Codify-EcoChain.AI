from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# This exact line must be present:
app = FastAPI(title="EcoChain Climate AI API")

# CRUCIAL: Cross-Origin Resource Sharing (CORS) allows your frontend to talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any local frontend port (React, Live Server, etc.) to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your newly trained pipeline weights
model = joblib.load("climate_model_pipeline.pkl")

# Strict schema matching your exact carbon dataset features
class ClimateInputSchema(BaseModel):
    Country_code: str
    Country_name: str
    Series_code: str
    Series_name: str
    SCALE: float
    Decimals: float
    year_1990: float
    year_1995: float
    year_2000: float
    year_2005: float
    year_2010: float

@app.post("/predict")
def predict_emissions(payload: ClimateInputSchema):
    # Map API keys to the exact DataFrame column headers your model expects
    input_data = {
        'Country code': payload.Country_code,
        'Country name': payload.Country_name,
        'Series code': payload.Series_code,
        'Series name': payload.Series_name,
        'SCALE': payload.SCALE,
        'Decimals': payload.Decimals,
        '1990': payload.year_1990,
        '1995': payload.year_1995,
        '2000': payload.year_2000,
        '2005': payload.year_2005,
        '2010': payload.year_2010
    }
    
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    
    return {
        "success": True,
        "predicted_emissions_2011": float(prediction[0])
    }