
from dependencies import  mp

# Configurar Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, 
        results.face_landmarks
        )
    mp_drawing.draw_landmarks(
        image, 
        results.pose_landmarks, 
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec( color = ( 80, 22, 10 ), thickness = 2, circle_radius= 2 ),
        mp_drawing.DrawingSpec( color = ( 80, 44, 121 ), thickness = 2, circle_radius= 2 ),
        )
    mp_drawing.draw_landmarks(
        image, 
        results.left_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec( color = ( 121, 22, 76 ), thickness = 2, circle_radius= 4 ),
        mp_drawing.DrawingSpec( color = ( 121, 44, 250 ), thickness = 2, circle_radius= 2 ),
        )
    mp_drawing.draw_landmarks(
        image, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec( color = ( 245, 117, 66 ), thickness = 2, circle_radius= 4 ),
        mp_drawing.DrawingSpec( color = ( 245, 66, 230 ), thickness = 2, circle_radius= 2 ),
        )