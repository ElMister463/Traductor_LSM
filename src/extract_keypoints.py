from dependencies import  mp, np
#from draw_landmarks import draw_landmarks 

# Configurar Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def extract_keypoints( results ):
    pose = np.array ([[res.x, res.y, res.z, res.visibility ] 
        for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros( 33*4 )
    
    face = np.array ([[res.x, res.y, res.z ] 
        for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros( 468*3 )

    lh = np.array ([[res.x, res.y, res.z ] 
        for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros( 21*3 )

    rh = np.array ([[res.x, res.y, res.z ] 
        for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros( 21*3 )

    return np.concatenate( [ pose, face, lh, rh ] )

    print(extract_keypoints( results ).shape)