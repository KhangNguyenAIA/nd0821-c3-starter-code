# Put the code for your API here.
from fastapi import Body, FastAPI
import pandas as pd
import joblib
import json
from pydantic import BaseModel, Field

from starter.ml.model import inference, process_data

import os
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    print("Running DVC")
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

class Value(BaseModel):
    age: int = Field(None, example=39)
    workclass: str = Field(None, example="State-gov")
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example="Bachelors")
    education_num: int = Field(None, alias='education-num', example=13)
    marital_status: str = Field(None, 
                                alias='marital-status', 
                                example="Never-married")
    occupation: str = Field(None, example="Adm-clerical")
    relationship: str = Field(None, example="Not-in-family")
    race: str = Field(None, example="White")
    sex: str = Field(None, example="Male")
    capital_gain: int = Field(None, alias='capital-gain', example=2174)
    capital_loss: int = Field(None, alias='capital-loss', example=0)
    hours_per_week: int = Field(None, alias='hours-per-week', example=0)
    native_country: str = Field(None, 
                                alias='native-country', 
                                example="United-States")

    class Config:
        schema_extra = {
            'examples': {
                "normal": {
                    "summary": "A normal example",
                    "description": "API workds well",
                    "value": {
                        "age": 39,
                        "workclass": "State-gov",
                        "fnlgt": 77516,
                        "education": "Bachelors",
                        "education-num": 13,
                        "marital-status": "Never-married",
                        "occupation": "Adm-clerical",
                        "relationship": "Not-in-family",
                        "race": "White",
                        "sex": "Male",
                        "capital-gain": 2174,
                        "capital-loss": 0,
                        "hours-per-week": 40,
                        "native-country": "United-States"
                    }
                },
                "invalid": {
                    "summary": "An invalid example",
                    "description": "API will fail",
                    "value": {
                        "age": '39',
                        "workclass": "State-gov",
                        "fnlgt": 77516,
                        "education": "Bachelors",
                        "education-num": 13,
                        "marital-status": "Never-married",
                        "occupation": "Adm-clerical",
                        "relationship": "Not-in-family",
                        "race": "White",
                        "sex": "Male",
                        "capital-gain": 2174,
                        "capital-loss": 0,
                        "hours-per-week": '40',
                        "native-country": "United-States"
                    },
                },
            }
        }



cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]
# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

@app.post("/inference")
async def get_inference(body: Value = Body(
    ...,
    examples=Value.Config.schema_extra['examples']
)):
    data = pd.DataFrame([{"age" : body.age,
                        "workclass" : body.workclass,
                        "fnlgt" : body.fnlgt,
                        "education" : body.education,
                        "education-num" : body.education_num,
                        "marital-status" : body.marital_status,
                        "occupation" : body.occupation,
                        "relationship" : body.relationship,
                        "race" : body.race,
                        "sex" : body.sex,
                        "capital-gain" : body.capital_gain,
                        "capital-loss" : body.capital_loss,
                        "hours-per-week" : body.hours_per_week,
                        "native-country" : body.native_country}])
    lr_model = joblib.load("./model/model.pkl") 
    enc = joblib.load("./model/encoder.enc")
    lb = joblib.load("./model/lb.enc")

    X, _, _, _ = process_data(data, 
                            categorical_features=cat_features, 
                            training=False, 
                            encoder = enc, 
                            lb = lb) 

    preds = inference(lr_model, X)
    if preds[0] == 1:
        prediction = "Salary > 50k"
    else:
        prediction = "Salary <= 50k"
    return {"result": prediction}