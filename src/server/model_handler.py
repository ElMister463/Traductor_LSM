import tensorflow as tf
import numpy as np
import os


def load_model(model_path=None):
    """
    Carga el modelo TFLite desde el archivo.
    Si no se proporciona un 'model_path', se carga desde la variable de entorno.
    """
    if model_path is None:
        model_path = 'model.tflite'  # Cambia a una ruta relativa

    
    if model_path is None:
        raise ValueError("No se ha proporcionado el 'MODEL_PATH' en el archivo .env o como argumento.")
    
    # Verifica que el archivo de modelo exista
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El archivo del modelo no se encuentra en la ruta: {model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def make_prediction(interpreter, input_data):
    """
    Realiza una predicción utilizando el modelo TFLite.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Convertir los datos de entrada al formato esperado
    input_array = np.array(input_data, dtype=np.float32).reshape(input_details[0]['shape'])
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()

    # Obtener la predicción del modelo
    output = interpreter.get_tensor(output_details[0]['index'])
    return output.tolist()
