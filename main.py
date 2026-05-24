from fastapi import FastAPI, HTTPException
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


@app.post("/predict_future")
def predict_future_trajectory(payload: ClimateInputSchema, target_year: int):
    try:
        if target_year < 2011:
            raise HTTPException(
                status_code=400, 
                detail="Target year must be 2011 or later for future forecasting."
            )

        # 1. Create a deep local copy of the historical data to manipulate safely
        current_history = {
            '1990': float(payload.year_1990),
            '1995': float(payload.year_1995),
            '2000': float(payload.year_2000),
            '2005': float(payload.year_2005),
            '2010': float(payload.year_2010)
        }
        
        last_computed_prediction = 0.0
        
        # 2. Mathematically map how many 5-year leaps we need to perform
        # 2011 -> 1 loop | 2015 -> 1 loop | 2020 -> 2 loops | 2025 -> 3 loops
        current_year = 2011
        while current_year <= target_year:
            
            # Format the exact input payload for this step's decision matrix
            step_input = {
                'Country code': payload.Country_code,
                'Country name': payload.Country_name,
                'Series code': payload.Series_code,
                'Series name': payload.Series_name,
                'SCALE': payload.SCALE,
                'Decimals': payload.Decimals,
                '1990': current_history['1990'],
                '1995': current_history['1995'],
                '2000': current_history['2000'],
                '2005': current_history['2005'],
                '2010': current_history['2010']
            }
            
            # Compute inference for this time block
            df_step = pd.DataFrame([step_input])
            last_computed_prediction = float(model.predict(df_step)[0])
            
            # Shift the lookback window frames forward explicitly by 5 years
            current_history['1990'] = current_history['1995']
            current_history['1995'] = current_history['2000']
            current_history['2000'] = current_history['2005']
            current_history['2005'] = current_history['2010']
            current_history['2010'] = last_computed_prediction
            
            # Move our iterator forward to the next cycle block
            if current_year == 2011:
                current_year = 2015  # First step snaps up to our 5-year grid
            else:
                current_year += 5
                
        return {
            "success": True,
            "target_year": target_year,
            "predicted_emissions": round(last_computed_prediction, 2)
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))