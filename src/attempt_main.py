import cv2
from dependencies import mp, np, os, draw_landmarks, mediapipe_detection, extract_keypoints, prepare_data
import model_training

# Configurar Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

DATA_PATH = 'MP_Data'  # Ruta donde están guardados los datos

actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])

# Función para visualizar probabilidades
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Función para verificar si las manos están presentes
def hands_present(results):
    return results.left_hand_landmarks or results.right_hand_landmarks

# Cargar el modelo entrenado
model = model_training.load_trained_model()

sequence = []
sentence = []
predictions = []
threshold = 0.75  # Este es el umbral, en cuanto mas aumentes, mayor precisio, pero más lento

# Configuración de la cámara
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

        # Verificar si hay manos presentes antes de predecir
        if hands_present(results):
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Mantener solo los últimos 30 frames

            if len(sequence) == 30:  # Realizar la predicción solo con una secuencia completa
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                # Verificar la estabilidad de la predicción
                if len(predictions) > 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0 and actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                        elif len(sentence) == 0:
                            sentence.append(actions[np.argmax(res)])

                    # Mantener solo las últimas 5 palabras en la frase
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
