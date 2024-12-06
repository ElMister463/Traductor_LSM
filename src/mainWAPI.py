import cv2
from dependencies import mp, np, os, draw_landmarks, mediapipe_detection, extract_keypoints
import model_training
import requests
import tensorflow as tf
from collections import deque
import time

# Configurar Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (200, 123, 211)]

DATA_PATH = "C:/Users/ricar/Desktop/Traductor_LSM/MP_Data" 
actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])



# Cargar el modelo TFLite
interpreter = model_training.load_trained_model(model_path='model.tflite')

sequence = []
sentence = []
predictions = []
smooth_predictions = deque(maxlen=10)
threshold = 0.73

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Poner el modelo Mediapipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    no_hands_detected = True  # Variable para controlar el estado de detección de manos
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detección con Mediapipe
        image, results = mediapipe_detection(frame, holistic)

        # Dibujar landmarks
        draw_landmarks(image, results)

        # Verificar si se detectan manos
        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
                input_details = interpreter.get_input_details()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_details = interpreter.get_output_details()
                res = interpreter.get_tensor(output_details[0]['index'])[0]

                smooth_predictions.append(np.argmax(res))
                most_common_prediction = np.bincount(smooth_predictions).argmax()
                predicted_action = actions[most_common_prediction]
                predictions.append(most_common_prediction)

                # Enviar la predicción a la API
                if res[most_common_prediction] > threshold:
                    response = requests.post('http://127.0.0.1:8000/predict', json={"data": keypoints.tolist()})
                    if response.status_code == 200:
                        api_response = response.json()
                        print(f"Respuesta de la API: {api_response}")

                # Acumular acciones en la oración sin repeticiones
                if predicted_action not in sentence:
                    sentence.append(predicted_action)

            no_hands_detected = False  # Se detectaron manos

        else:
            # Si no se detectan manos, formar la oración
            if sentence:
                formed_sentence = ' '.join(sentence)
                print(f"Oración formada: {formed_sentence}")
                sentence.clear()  # Limpiar la oración después de formarla
            else:
                if not no_hands_detected:  # Solo mostrar el mensaje una vez
                    print("No se detectan manos, esperando...")
                    no_hands_detected = True  # Actualizar el estado

        # Mostrar el frame procesado
        cv2.imshow('Traductor de Lenguaje de Señas', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()