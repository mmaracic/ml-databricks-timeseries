"""Module for calculating quality metrics for forecasting models."""
import numpy as np
from pydantic import BaseModel


class Metrics(BaseModel):
    store: int
    product: int
    mse: float
    mae: float
    rmse: float

"""Calculate quality metrics for a forecasting model."""
def calculate_metrics(store, product, model, actual):
    pred = model.fittedvalues.values[:len(actual)]
    mse = np.mean((actual - pred) ** 2)
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(mse)
    return Metrics(
        store=store,
        product=product,
        mse=mse,
        mae=mae,
        rmse=rmse
    )
