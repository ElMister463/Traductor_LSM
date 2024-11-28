import cv2
from dependencies import mp, np, os, draw_landmarks, mediapipe_detection, extract_keypoints
import model_training
import requests
import tensorflow as tf
from collections import deque
import time  # Importar el módulo time para manejar el tiempo

# Configurar Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (200, 123, 211)]

DATA_PATH = 'MP_DATA'
actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Cargar el modelo TFLite
interpreter = model_training.load_trained_model(model_path='model.tflite')

sequence = []
sentence = []
predictions = []
smooth_predictions = deque(maxlen=10)  # Para suavizar las predicciones
threshold = 0.75

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables para controlar la frecuencia de impresión
last_print_time = time.time()
print_interval = 1.0  # Intervalo de tiempo en segundos para imprimir la acción

# Poner el modelo Mediapipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:  # Iniciar un bucle para la captura de video
        ret, frame = cap.read()
        if not ret:
            break  # Salir del bucle si no se puede capturar el frame

        # Detección con Mediapipe
        image, results = mediapipe_detection(frame, holistic)

        # Dibujar landmarks
        draw_landmarks(image, results)

        # Verificar si se detectan manos
        if results.left_hand_landmarks or results.right_hand_landmarks:
            # Predicción
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Mantener solo los últimos 30 frames

            if len(sequence) == 30:  # Cambié la condición a 30 frames
                # Prepara los datos de entrada
                input_data = np.expand_dims(sequence, axis=0).astype(np.float32)

                # Establece el tensor de entrada
                input_details = interpreter.get_input_details()
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # Ejecuta la inferencia
                interpreter.invoke()

                # Obtiene la predicción
                output_details = interpreter.get_output_details()
                res = interpreter.get_tensor(output_details[0]['index'])[0]

                smooth_predictions.append(np.argmax(res))
                most_common_prediction = np.bincount(smooth_predictions).argmax()
                predicted_action = actions[most_common_prediction]
                predictions.append(most_common_prediction)

                # Enviar la predicción a la API
                if res[most_common_prediction] > threshold:  # Solo enviar si la confianza es alta
                    response = requests.post('http://127.0.0.1:8000/predict', json={"data": keypoints.tolist()})
                    if response.status_code == 200:
                        api_response = response.json()
                        print(f"Respuesta de la API: {api_response}")

                # Lógica de visualización
            if len(predictions) > 10 and np.unique(predictions[-10:]).size == 1:
                if time.time() - last_print_time > print_interval:
                    print(f"Predicción actual: {actions[np.unique(predictions[-10:])[0]]}")
                    last_print_time = time.time()
        else:
            # Si no se detectan manos, no hacer nada
            print("No se detectan manos, esperando...")

        # Mostrar el frame procesado
        cv2.imshow('Hand Gesture Recognition', image)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()