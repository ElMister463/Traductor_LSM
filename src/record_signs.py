import cv2

from dependencies import  mp, np, os, draw_landmarks, mediapipe_detection, extract_keypoints, prepare_data
#from draw_landmarks import draw_landmarks 

# Configurar Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic



# Path para exportar la data, y los numpy arrays
DATA_PATH = os.path.join( 'MP_Data' )

# Señas o Acciones que se van a detectar
actions = np.array([ 'adios' ])

# Treinta videoes por data
no_sequences = 30

# Videos que será de 30 frames
sequence_length = 30

# Generar carpetas por cada seña
for action in actions:
    for sequence in range ( no_sequences ):
        try:
            os.makedirs( os.path.join( DATA_PATH, action, str( sequence ) ) )
        except:
            pass


# cap = cv2.VideoCapture(config('IP_WEBCAM'))
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Poner el modelo Mediapipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:

        for sequence in range ( no_sequences ):

            for frame_num in range ( sequence_length ):

                # Leer el feed
                ret, frame = cap.read()

                # Verificar si la captura de video fue exitosa
                if not ret:
                    print("Error: No se pudo obtener la imagen de la cámara")
                    continue

                # Detección con Mediapipe
                image, results = mediapipe_detection(frame, holistic)

                # Dibujar landmarks
                draw_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText( image, 'INICIANDO', ( 120, 200 ),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, ( 0, 255, 0 ), 4, cv2.LINE_AA )
                    cv2.putText( image, 'Grabando {} Numero de video {}'.format( action, sequence ), ( 15, 12 ),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 1, cv2.LINE_AA )

                    cv2.imshow('Traductor LSM', image)
                    cv2.waitKey( 2000 )
                
                else:
                    cv2.putText( image, 'Grabando {} Numero de video {}'.format( action, sequence ), ( 15, 12 ),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 1, cv2.LINE_AA )
                    
                    cv2.imshow('Traductor LSM', image)

                keypoints = extract_keypoints( results )
                npy_path = os.path.join( DATA_PATH, action, str( sequence ), str( frame_num ))
                np.save( npy_path, keypoints )

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

# Preparar datos
X_train, X_test, y_train, y_test = prepare_data(actions, DATA_PATH, no_sequences, sequence_length)


