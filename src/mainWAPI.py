import cv2
from dependencies import mp, np, os, draw_landmarks, mediapipe_detection, extract_keypoints, load_dotenv
import model_training
import requests
import tensorflow as tf
from collections import deque

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

        # Predicción
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Aumentar a 60 frames

        if len(sequence) == 30:  # Cambié la condición a 60 frames
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
            print(predicted_action)
            predictions.append(most_common_prediction)

            # Enviar la predicción a la API
            if res[most_common_prediction] > threshold:  # Solo enviar si la confianza es alta
                response = requests.post('http://127.0.0.1:8000/predict', json={"data": keypoints.tolist()})
                if response.status_code == 200:
                    api_response = response.json()
                    print(f"Respuesta de la API: {api_response}")

        # Lógica de visualización
        if len(predictions) > 10 and np.unique(predictions[-10:])[0] == most_common_prediction:
            if len(sentence) > 0 and actions[most_common_prediction] != sentence[-1]:
                sentence.append(actions[most_common_prediction])
            elif len(sentence) == 0:
                sentence.append(actions[most_common_prediction])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Visualizar probabilidades
            image = prob_viz(res, actions, image, colors)
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Traductor LSM', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()