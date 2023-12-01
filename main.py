import pickle
from typing import List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


app = FastAPI()


@app.get("/favicon.ico")
async def get_favicon():
    return FileResponse("path_to_your_favicon.ico")


@app.get("/")
def root():
    return "This is my ML model"


@app.post("/predict_item")
async def predict_item(item: Item):
    input_data = {
        "name": [item.name],
        "year": [item.year],
        "selling_price": [item.selling_price],
        "km_driven": [item.km_driven],
        "fuel": [item.fuel],
        "seller_type": [item.seller_type],
        "transmission": [item.transmission],
        "owner": [item.owner],
        "mileage": [item.mileage],
        "engine": [item.engine],
        "max_power": [item.max_power],
        "torque": [item.torque],
        "seats": [item.seats]
    }
    input_df = pd.DataFrame(input_data)
    input_df_encoded = encoder.transform(input_df[['fuel', 'seller_type', 'transmission', 'owner']])
    input_features = pd.concat([input_df_encoded, input_df[['year', 'selling_price', 'km_driven', 'seats']]], axis=1)
    prediction = model.predict(input_features)
    return {"prediction": prediction[0]}


@app.post("/predict_items")
async def predict_items(items: Items):
    predictions = []
    for item in items.objects:
        prediction = predict_item(item)
        predictions.append(prediction)
    return {"predictions": predictions}
