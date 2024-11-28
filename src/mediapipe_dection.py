import cv2
from dependencies import mp

# Configurar Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    # Convertir y procesar imagen
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model_output = model.process(rgb_image)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), model_output


'''
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results'''