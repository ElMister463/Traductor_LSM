
from pydantic import BaseModel
from typing import List

class InputData ( BaseModel ):
    """ 
    Estructura de los datos de entrada
    """
    data: List[ float ] #Los datos deben ser una lista de numeros