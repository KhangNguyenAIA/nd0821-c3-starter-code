# Put the code for your API here.
from fastapi import FastAPI
import pandas as pd
import joblib
import json
from pydantic import BaseModel, Field

from starter.ml.model import inference, process_data

import os
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    print("Running DVC")
    os.system("dvc config cache.type copy")
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

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
async def get_inference(body: Value):
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