import cv2
import os
import numpy as np
from dependencies import mp, draw_landmarks, mediapipe_detection, extract_keypoints

# Configurar Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Path para exportar la data, y los numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Señas o Acciones que se van a detectar
actions = np.array(['adios'])

# Treinta videos por data
no_sequences = 30

# Videos que serán de 60 frames
sequence_length = 30

# Generar carpetas por cada seña
for action in actions:
    for sequence in range(no_sequences):
        dir_path = os.path.join(DATA_PATH, action, str(sequence))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# Configuración de la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara")
    exit()

# Configuración de Mediapipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("Error: No se pudo obtener la imagen de la cámara")
                    continue

                # Detección con Mediapipe
                image, results = mediapipe_detection(frame, holistic)

                # Dibujar los landmarks
                draw_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'INICIANDO', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Grabando {} Numero de video {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow('Traductor LSM', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Grabando {} Numero de video {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Traductor LSM', image)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
