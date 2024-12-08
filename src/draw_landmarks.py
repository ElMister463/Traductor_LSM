
from dependencies import  mp

# Configurar Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def draw_landmarks(image, results):
    connections = {
        "face": (results.face_landmarks, None),
        "pose": (results.pose_landmarks, mp_holistic.POSE_CONNECTIONS),
        "left_hand": (results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS),
        "right_hand": (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    }

    specs = {
        "pose": (mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                 mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)),
        "left_hand": (mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)),
        "right_hand": (mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    }

    for key, (landmarks, conn) in connections.items():
        if landmarks:
            if conn:
                spec = specs.get(key, (None, None))
                mp_drawing.draw_landmarks(image, landmarks, conn, *spec)
            else:
                mp_drawing.draw_landmarks(image, landmarks)
