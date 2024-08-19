from flask import Blueprint, jsonify, request, Response
import cv2
import mediapipe as mp
import numpy as np
import threading
import base64
import os
from io import BytesIO
from PIL import Image

abecedario_api = Blueprint('abecedario_api', __name__)

# Inicialización de MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
carpeta_imagenes = os.path.join(BASE_DIR, 'abc')

# Asegúrate de que las imágenes se carguen correctamente
imagenes_letras = {}


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
    
    ring_finger_pip2 = (int(hand_landmarks.landmark[5].x * image_width),
                                int(hand_landmarks.landmark[5].y * image_height))


    if thumb_tip[1] < index_finger_tip[1] and thumb_tip[1] < middle_finger_tip[1] and thumb_tip[1] < ring_finger_tip[1] and thumb_tip[1] < pinky_tip[1]:
        return 'A'
    elif index_finger_pip[1] - index_finger_tip[1]>0 and pinky_pip[1] - pinky_tip[1] > 0 and \
        middle_finger_pip[1] - middle_finger_tip[1] >0 and ring_finger_pip[1] - ring_finger_tip[1] >0 and \
            middle_finger_tip[1] - ring_finger_tip[1] <0 and abs(thumb_tip[1] - ring_finger_pip2[1])<40:
        return 'B'
    elif (distancia_euclidiana(thumb_tip, middle_finger_tip) < 65 and 
          distancia_euclidiana(thumb_tip, ring_finger_tip) < 65 and 
          pinky_pip[1] - pinky_tip[1] < 0 and 
          index_finger_pip[1] - index_finger_tip[1] > 0):
        return 'D'
    elif abs(index_finger_tip[1] - thumb_tip[1]) < 380 and \
        index_finger_tip[1] - middle_finger_pip[1]<0 and index_finger_tip[1] - middle_finger_tip[1] < 0 and \
            index_finger_tip[1] - index_finger_pip[1] > 0:
        return "C"
    elif index_finger_pip[1] - index_finger_tip[1] < 0 and pinky_pip[1] - pinky_tip[1] < 0 and \
        middle_finger_pip[1] - middle_finger_tip[1] < 0 and ring_finger_pip[1] - ring_finger_tip[1] < 0 \
            and abs(index_finger_tip[1] - thumb_tip[1]) > 100 and \
                thumb_tip[1] - index_finger_tip[1] > 0 \
                and thumb_tip[1] - middle_finger_tip[1] > 0 \
                and thumb_tip[1] - ring_finger_tip[1] > 0 \
                and thumb_tip[1] - pinky_tip[1] > 0:
        return 'E'
    elif (pinky_pip[1] - pinky_tip[1] > 0 and 
          middle_finger_pip[1] - middle_finger_tip[1] > 0 and 
          ring_finger_pip[1] - ring_finger_tip[1] > 0 and 
          index_finger_pip[1] - index_finger_tip[1] < 0 and 
          abs(thumb_pip[1] - thumb_tip[1]) > 0 and 
          distancia_euclidiana(index_finger_tip, thumb_tip) < 65):
        return 'F'
    elif (index_finger_tip[1] < thumb_tip[1] and
          index_finger_tip[1] < middle_finger_tip[1] and
          index_finger_tip[1] < ring_finger_tip[1] and
          index_finger_tip[1] < pinky_tip[1] and
          thumb_pip[1] - thumb_tip[1] < 0 and
          middle_finger_tip[1] - middle_finger_pip[1] > 0 and
          ring_finger_tip[1] - ring_finger_pip[1] > 0 and
          pinky_tip[1] - pinky_pip[1] > 0):
        return 'G'
    elif (index_finger_tip[1] < middle_finger_tip[1] and
          index_finger_tip[1] < ring_finger_tip[1] and
          index_finger_tip[1] < pinky_tip[1] and
          middle_finger_tip[1] < ring_finger_tip[1] and
          middle_finger_tip[1] < pinky_tip[1] and
          thumb_pip[1] - thumb_tip[1] < 0 and
          ring_finger_tip[1] - ring_finger_pip[1] > 0 and
          pinky_tip[1] - pinky_pip[1] > 0):
        return 'H'
    elif (pinky_tip[1] < thumb_tip[1] and
        pinky_tip[1] < index_finger_tip[1] and
        pinky_tip[1] < middle_finger_tip[1] and
        pinky_tip[1] < ring_finger_tip[1] and
        pinky_tip[1] < pinky_pip[1] and
        index_finger_pip[1] - index_finger_tip[1] < 10 and
        middle_finger_pip[1] - middle_finger_tip[1] < 10 and
        ring_finger_pip[1] - ring_finger_tip[1] < 10 and
        thumb_tip[1] - thumb_pip[1] < 10):
        return 'I'
    elif (index_finger_tip[1] < thumb_tip[1] and
        index_finger_tip[1] < middle_finger_tip[1] and
        index_finger_tip[1] < ring_finger_tip[1] and
        index_finger_tip[1] < pinky_tip[1] and
        middle_finger_tip[1] < ring_finger_tip[1] and
        middle_finger_tip[1] < pinky_tip[1] and
        abs(thumb_tip[1] - thumb_pip[1]) < 30 and
        abs(ring_finger_tip[1] - ring_finger_pip[1]) < 30 and
        abs(pinky_tip[1] - pinky_pip[1]) < 30):
        return 'K'
    elif distancia_euclidiana(thumb_tip, middle_finger_tip) > 190 \
        and distancia_euclidiana(thumb_tip, ring_finger_tip) > 190 \
        and  pinky_pip[1] - pinky_tip[1]<0\
        and index_finger_pip[1] - index_finger_tip[1]>0:
        return 'L'
    elif index_finger_pip[1] - index_finger_tip[1] < 0 and pinky_pip[1] - pinky_tip[1] < 0 and \
        middle_finger_pip[1] - middle_finger_tip[1] < 0 and ring_finger_pip[1] - ring_finger_tip[1] < 0 \
            and abs(index_finger_tip[1] - thumb_tip[1]) < 25 and \
                thumb_tip[1] - index_finger_tip[1] > 0 \
                and thumb_tip[1] - middle_finger_tip[1] > 0 \
                and thumb_tip[1] - ring_finger_tip[1] > 0 \
                and thumb_tip[1] - pinky_tip[1] > 0:
        return 'M'
    elif distancia_euclidiana(thumb_tip, middle_finger_tip) < 100 \
        and distancia_euclidiana(thumb_tip, ring_finger_tip) < 120 \
        and  pinky_pip[1] - pinky_tip[1]<0\
        and index_finger_pip[1] - index_finger_tip[1]<0:
        return 'O'
    elif (index_finger_tip[1] < thumb_tip[1] and
        index_finger_tip[1] < middle_finger_tip[1] and
        index_finger_tip[1] < ring_finger_tip[1] and
        index_finger_tip[1] < pinky_tip[1] and
        middle_finger_pip[1] < middle_finger_tip[1] and
        ring_finger_pip[1] < ring_finger_tip[1] and
        pinky_pip[1] < pinky_tip[1] and
        abs(thumb_tip[0] - index_finger_pip[0]) < 30):
        return 'P'
# Ruta para detectar gestos
@abecedario_api.route('/detectar_abecedario', methods=['POST'])
def detectar_abecedario():
    data = request.get_json()
    image_base64 = data.get('image')
    
    if image_base64:
        # Decodificar la imagen desde base64
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        height, width, _ = image.shape
        new_width = 640
        new_height = int((new_width / width) * height)
        image = cv2.resize(image, (new_width, new_height))

        # Procesar la imagen con MediaPipe Hands
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_bounding_box(image, hand_landmarks)
                gesture = procesar_gesto(hand_landmarks, image)
                print("Gesto detectado:", gesture)

                gesture_image_base64 = None
                # Obtener la imagen del gesto
                if gesture in imagenes_letras:
                    letra_image = imagenes_letras[gesture]
                    letra_image_resized = cv2.resize(letra_image, (50, 50))
                    x_offset, y_offset = 10, 10
                    if x_offset + letra_image_resized.shape[1] <= image.shape[1] and y_offset + letra_image_resized.shape[0] <= image.shape[0]:
                        image[y_offset:y_offset + letra_image_resized.shape[0], x_offset:x_offset + letra_image_resized.shape[1]] = letra_image_resized
                
                # Dibujar las marcas de la mano
                #mp_drawing.draw_landmarks(
                    #image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    #mp_drawing_styles.get_default_hand_landmarks_style(),
                    #mp_drawing_styles.get_default_hand_connections_style())
            
            # Codificar la imagen de vuelta a base64
            _, buffer = cv2.imencode('.png', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({"image": image_base64, "gesture": gesture})
        else:
            return jsonify({"gesture": "No se detectaron manos"})
    
    return Response(response='Imagen no válida', status=400)

if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(abecedario_api)
    app.run(host='0.0.0.0', port=5000, debug=True)
