from flask import Flask, Blueprint, Response
import cv2
import mediapipe as mp
import numpy as np
import threading

palabras_api = Blueprint('palabras_api', __name__)

# Inicialización de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(p1, p2):
    d = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    return d

# Función para dibujar el cuadro delimitador alrededor de la mano detectada
def draw_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0
    
    # Iterar a través de los landmarks para encontrar las coordenadas del cuadro delimitador
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y
    
    # Dibujar el cuadro delimitador
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Función para procesar los gestos y devolver el gesto detectado
def procesar_gesto(hand_landmarks, image):
    image_height, image_width, _ = image.shape

    thumb_tip = (int(hand_landmarks.landmark[4].x * image_width),
                 int(hand_landmarks.landmark[4].y * image_height))
    index_finger_tip = (int(hand_landmarks.landmark[8].x * image_width),
                        int(hand_landmarks.landmark[8].y * image_height))
    index_finger_pip = (int(hand_landmarks.landmark[6].x * image_width),
                        int(hand_landmarks.landmark[6].y * image_height))
    thumb_pip = (int(hand_landmarks.landmark[2].x * image_width),
                 int(hand_landmarks.landmark[2].y * image_height))

    middle_finger_tip = (int(hand_landmarks.landmark[12].x * image_width),
                         int(hand_landmarks.landmark[12].y * image_height))
    middle_finger_pip = (int(hand_landmarks.landmark[10].x * image_width),
                         int(hand_landmarks.landmark[10].y * image_height))

    ring_finger_tip = (int(hand_landmarks.landmark[16].x * image_width),
                       int(hand_landmarks.landmark[16].y * image_height))
    ring_finger_pip = (int(hand_landmarks.landmark[14].x * image_width),
                       int(hand_landmarks.landmark[14].y * image_height))

    pinky_tip = (int(hand_landmarks.landmark[20].x * image_width),
                 int(hand_landmarks.landmark[20].y * image_height))
    pinky_pip = (int(hand_landmarks.landmark[18].x * image_width),
                 int(hand_landmarks.landmark[18].y * image_height))

    # Detectar gestos basados en las posiciones de los puntos clave
    if (abs(thumb_tip[1] - index_finger_pip[1]) < 45 and
        abs(thumb_tip[1] - middle_finger_pip[1]) < 30 and
        abs(thumb_tip[1] - ring_finger_pip[1]) < 30 and
        abs(thumb_tip[1] - pinky_pip[1]) < 30):
        return 'A'

    elif (index_finger_pip[1] - index_finger_tip[1] > 0 and
          pinky_pip[1] - pinky_tip[1] > 0 and
          middle_finger_pip[1] - middle_finger_tip[1] < 0 and
          ring_finger_pip[1] - ring_finger_tip[1] < 0 and
          abs(index_finger_tip[1] - thumb_tip[1]) < 360 and
          thumb_tip[1] - index_finger_tip[1] > 0 and
          thumb_tip[1] - middle_finger_tip[1] > 0 and
          thumb_tip[1] - ring_finger_tip[1] > 0 and
          thumb_tip[1] - pinky_tip[1] > 0):
        return 'Te quiero'

    elif (thumb_tip[0] > index_finger_tip[0] and
          thumb_tip[0] > middle_finger_tip[0] and
          thumb_tip[0] > ring_finger_tip[0] and
          thumb_tip[0] > pinky_tip[0] and
          index_finger_pip[0] < index_finger_tip[0] and
          pinky_pip[0] < pinky_tip[0] and
          middle_finger_pip[0] < middle_finger_tip[0] and
          ring_finger_pip[0] < ring_finger_tip[0] and
          abs(index_finger_tip[0] - thumb_tip[0]) < 360):
        return 'Bien'

    elif (thumb_tip[0] < index_finger_tip[0] and
          thumb_tip[0] < middle_finger_tip[0] and
          thumb_tip[0] < ring_finger_tip[0] and
          thumb_tip[0] < pinky_tip[0] and
          index_finger_pip[0] > index_finger_tip[0] and
          pinky_pip[0] > pinky_tip[0] and
          middle_finger_pip[0] > middle_finger_tip[0] and
          ring_finger_pip[0] > ring_finger_tip[0] and
          abs(index_finger_tip[0] - thumb_tip[0]) < 360):
        return 'Mal'

    return None

# Función para procesar el video y detectar gestos en tiempo real
def procesar_video():
    global cap

    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Draw bounding box
                    draw_bounding_box(image, hand_landmarks)

                    # Procesar el gesto detectado
                    gesture = procesar_gesto(hand_landmarks, image)
                    if gesture:
                        cv2.putText(image, gesture, (700, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    3.0, (0, 0, 255), 6)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()

# Ruta para iniciar el procesamiento del video en tiempo real
@palabras_api.route('/detectar_gestos', methods=['GET'])
def detectar_gestos():
    global video_thread
    if 'video_thread' not in globals() or not video_thread.is_alive():
        video_thread = threading.Thread(target=procesar_video)
        video_thread.start()
        return Response(response='Procesamiento de video iniciado.', status=200)
    else:
        return Response(response='El procesamiento de video ya está en curso.', status=400)

# Ejecutar el servidor Flask
if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(palabras_api)
    app.run(host='0.0.0.0', port=5000, debug=True)
