import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .patientService import insert_patient

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / 'src'))
from inference import InputValidationError, validate_features


app= FastAPI()

# catch streamlit data
class request_body(BaseModel):
    bmi : float
    age : float
    genhlth : float
    income : float
    highbp : float
    highchol : float
    education : float

@app.get("/")
async def root():
    return {"Welcome Diabetes Prediction"}

@app.post("/predict")
async def predict(stlmdata: request_body):
    formData = stlmdata.model_dump()
    try:
        validate_features({
            "BMI": formData["bmi"], "Age": formData["age"], "GenHlth": formData["genhlth"],
            "Income": formData["income"], "HighBP": formData["highbp"],
            "HighChol": formData["highchol"], "Education": formData["education"],
        })
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.problems)
    values = tuple(formData.values())   # Convert dict to tuple
    if not insert_patient(values):
        raise HTTPException(status_code=503, detail="The record could not be saved.")
    return {"message": "Patient inserted successfully"}
