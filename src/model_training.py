import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# Configuración de la ruta de datos
DATA_PATH = 'MP_Data'  # Ruta donde están guardados los datos

# Obtener las acciones automáticamente de las carpetas
actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])

no_sequences = 30
sequence_length = 30

# Función para preparar los datos
def prepare_data(DATA_PATH, no_sequences, sequence_length):
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

# Función para cargar el modelo entrenado
def load_trained_model():
    return load_model('action.h5')

# Entrenamiento del modelo solo cuando se ejecuta directamente
if __name__ == "__main__":
    # Preparar los datos
    (X_train, X_test, y_train, y_test) = prepare_data(DATA_PATH, no_sequences, sequence_length)
    print("Preparación de datos completada.")

    # Configuración del TensorBoard
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    # Crear el modelo
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    # Compilar el modelo
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
    print("Entrenamiento del modelo completado.")

    # Guardar el modelo
    model.save('action.h5')
    print("Modelo guardado")
