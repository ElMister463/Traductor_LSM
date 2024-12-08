import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

import tensorflow as tf



# Configuración de la ruta de datos
DATA_PATH = "MP_Data"  # Ruta donde están guardados los datos

if DATA_PATH is None:
    raise ValueError("La variable de entorno DATA_PATH no está configurada. Asegúrate de tener un archivo .env correcto.")

# Obtener las acciones automáticamente de las carpetas
actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])

# Configuración de las secuencias
no_sequences = 30
sequence_length = 30

# Función para preparar los datos
def prepare_data(DATA_PATH, actions, no_sequences, sequence_length):
    """
    Prepara los datos para entrenamiento y prueba.

    Args:
        DATA_PATH (str): Ruta donde se encuentran los datos.
        actions (np.array): Lista de acciones.
        no_sequences (int): Número de secuencias por acción.
        sequence_length (int): Longitud de cada secuencia.

    Returns:
        X_train, X_test, y_train, y_test: Conjuntos de datos divididos para entrenamiento y prueba.
    """
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                res = np.load(npy_path)
                window.append(res)

            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(np.array(labels).astype(int))
    return train_test_split(X, y, test_size=0.05)

# Función para crear el modelo
def create_model(input_shape, num_classes):
    """
    Crea y compila un modelo LSTM.

    Args:
        input_shape (tuple): Forma de la entrada (tamaño de la secuencia y número de características).
        num_classes (int): Número de clases (acciones).

    Returns:
        model: Modelo LSTM compilado.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Salida con softmax para clasificación multiclase
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

# Función para entrenar el modelo
def train_model(model, X_train, y_train):
    """
    Entrena el modelo con los datos proporcionados.

    Args:
        model: El modelo compilado.
        X_train: Datos de entrenamiento.
        y_train: Etiquetas de entrenamiento.

    Returns:
        model: El modelo entrenado.
    """
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, epochs=300, callbacks=[tb_callback])
    return model

# Función para guardar el modelo
def save_model(model, model_path='action.h5'):
    """
    Guarda el modelo entrenado en el archivo especificado.

    Args:
        model: El modelo entrenado.
        model_path (str): Ruta donde se guardará el modelo.
    """
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

# Función para cargar el modelo previamente entrenado
# Cambia la función load_trained_model
def load_trained_model(model_path='model.tflite'):
    """
    Carga un modelo previamente entrenado en formato TFLite.

    Args:
        model_path (str): Ruta del modelo guardado.

    Returns:
        interpreter: El intérprete TFLite cargado.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Script principal
if __name__ == "__main__":
    # Preparar los datos
    print("Preparando los datos...")
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH, actions, no_sequences, sequence_length)
    print("Preparación de datos completada.")

    # Crear el modelo
    print("Creando el modelo...")
    input_shape = (sequence_length, X_train.shape[2])  # Asumimos que las características están en la 3ª dimensión
    model = create_model(input_shape, actions.shape[0])
    print("Modelo creado.")

    # Entrenar el modelo
    print("Entrenando el modelo...")
    model = train_model(model, X_train, y_train)
    print("Entrenamiento completado.")

    # Guardar el modelo
    save_model(model)
    print("¡Modelo entrenado y guardado exitosamente!")
