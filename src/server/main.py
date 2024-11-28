from fastapi import FastAPI
from model_handler import load_model, make_prediction
from schemas import InputData

app = FastAPI()

#Carga el modelo TFLite al iniciar el servior
model = load_model()

@app.post("/predict")
async def predict( input_data: InputData ):
    """ 
    Realiza una prediccion basada en los datos de entrada
    """

    result = make_prediction( model, input_data.data )
    return {"prediction": result}