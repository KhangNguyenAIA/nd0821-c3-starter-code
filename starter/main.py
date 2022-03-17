# Put the code for your API here.
from fastapi import FastAPI
import pandas as pd
import pickle
from typing import Dict, Union, List
from pydantic import BaseModel, Field

from starter.ml.model import inference

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

class Value(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

@app.post("/inference")
async def get_inference(body: Value):
    data = pd.DataFrame.from_dict(body)
    with open('../model/model.pkl', 'r') as file:
        lr_model = pickle.load(file)

    preds = inference(lr_model, data.to_numpy())
    return {"result": preds}