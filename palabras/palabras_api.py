from flask import Blueprint, jsonify, request, Response
import cv2
import mediapipe as mp
import numpy as np
import threading
import base64
from io import BytesIO
from PIL import Image

palabras_api = Blueprint('palabras_api', __name__)

# Inicialización de MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def distancia_euclidiana(p1, p2):
    d = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    return d

def draw_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0

    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

def procesar_gesto(hand_landmarks, image):
    image_height, image_width, _ = image.shape

    thumb_tip = (int(hand_landmarks.landmark[4].x * image_width),
                 int(hand_landmarks.landmark[4].y * image_height))
    thumb_pip = (int(hand_landmarks.landmark[3].x * image_width),
                 int(hand_landmarks.landmark[3].y * image_height))
    index_finger_tip = (int(hand_landmarks.landmark[8].x * image_width),
                        int(hand_landmarks.landmark[8].y * image_height))
    index_finger_pip = (int(hand_landmarks.landmark[6].x * image_width),
                        int(hand_landmarks.landmark[6].y * image_height))
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
    wrist = (int(hand_landmarks.landmark[0].x * image_width),
                int(hand_landmarks.landmark[0].y * image_height))
    ring_finger_pip2 = (int(hand_landmarks.landmark[5].x * image_width),
                                int(hand_landmarks.landmark[5].y * image_height))       

    # Detectar letras según el lenguaje de señas del Ecuador
    if (index_finger_pip[1] - index_finger_tip[1] > 0 and
        pinky_pip[1] - pinky_tip[1] > 0 and
        middle_finger_pip[1] - middle_finger_tip[1] < 0 and
        ring_finger_pip[1] - ring_finger_tip[1] < 0 and
        abs(index_finger_tip[1] - thumb_tip[1]) < 360 and
        thumb_tip[1] - index_finger_tip[1] > 0 and
        thumb_tip[1] - middle_finger_tip[1] > 0 and
        thumb_tip[1] - ring_finger_tip[1] > 0 and
        thumb_tip[1] - pinky_tip[1] > 0):
        return 'Te Quiero'
    elif (thumb_tip[1] < thumb_pip[1] and 
        index_finger_tip[1] > index_finger_pip[1] and 
        middle_finger_tip[1] > middle_finger_pip[1] and
        ring_finger_tip[1] > ring_finger_pip[1] and 
        pinky_tip[1] < pinky_pip[1]): 
        return 'Amigo'
    elif (thumb_tip[0] < index_finger_tip[0] and
        index_finger_tip[1] < thumb_tip[1] and
        middle_finger_tip[1] < ring_finger_tip[1]):
        return 'Casa'

# Ruta para detectar gestos
@palabras_api.route('/detectar_palabras', methods=['POST'])
def detectar_palabras():
    data = request.get_json()
    image_base64 = data.get('image')
    
    if image_base64:
        # Decodificar la imagen desde base64
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Procesar la imagen con MediaPipe Hands
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_bounding_box(image, hand_landmarks)
                word = procesar_gesto(hand_landmarks, image)
                print("Gesto detectado:", word)
                return jsonify({'word': word})
        else:
            return jsonify({'word': 'No se detectaron manos'})
    
    return Response(response='Imagen no válida', status=400)

if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(palabras_api)
    app.run(host='0.0.0.0', port=5000, debug=True)
