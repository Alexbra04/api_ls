from flask import Blueprint, jsonify, request, Response
import cv2
import mediapipe as mp
import numpy as np
import threading
import base64
import os
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
carpeta_imagenes = os.path.join(BASE_DIR, 'wrd')

def load_image_as_base64(image_name):
    image_path = os.path.join(carpeta_imagenes, image_name)
 # Verificar si el archivo existe
    if not os.path.isfile(image_path):
        print(f"El archivo {image_path} no existe.")
        return None

    try:
        with open(image_path, "rb") as image_file:
            # Codificar la imagen en base64
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_image
    except Exception as e:
        print(f"Error al cargar o codificar la imagen: {e}")
        return None


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
        palabra = 'Te Quiero'
        icono_base64 = load_image_as_base64('tequiero.png')
    elif (thumb_tip[0] < index_finger_tip[0] and
        index_finger_tip[1] < thumb_tip[1] and
        middle_finger_tip[1] < ring_finger_tip[1] and
        not (index_finger_tip[1] < thumb_tip[1] and
            thumb_tip[1] < index_finger_pip[1] and
            middle_finger_tip[1] < ring_finger_tip[1])):
        return 'Casa'
    elif (index_finger_tip[1] < thumb_tip[1] and
        thumb_tip[1] < index_finger_pip[1] and
        middle_finger_tip[1] < ring_finger_tip[1]):
        return 'Mamá'
    elif distancia_euclidiana(thumb_tip, middle_finger_tip) < 100 \
        and distancia_euclidiana(thumb_tip, ring_finger_tip) < 120 \
        and  pinky_pip[1] - pinky_tip[1]<0\
        and index_finger_pip[1] - index_finger_tip[1]>0:
        return 'Papá'
    elif(thumb_tip[0] < index_finger_tip[0] and
        index_finger_tip[1] > thumb_tip[1] and
        middle_finger_tip[1] < ring_finger_tip[1] and
        not (index_finger_tip[1] < thumb_tip[1] and
            thumb_tip[1] < index_finger_pip[1] and
            middle_finger_tip[1] < ring_finger_tip[1])):
        return 'Bien'

    return {'palabra': palabra, 'icono': icono_base64}

def rotar_imagen_a_vertical(image):
    # Obtener el alto y el ancho de la imagen
    height, width = image.shape[:2]
    
    # Si la imagen es más ancha que alta, rotar 90 grados
    if width > height:
        image = cv2.transpose(image)
        image = cv2.flip(image, flipCode=1)  # Flip horizontalmente
    return image


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
        
        # Asegurar que la imagen esté en orientación vertical
        image = rotar_imagen_a_vertical(image)

        # Procesar la imagen con MediaPipe Hands
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_bounding_box(image, hand_landmarks)
                word = procesar_gesto(hand_landmarks, image)
                print("Gesto detectado:", word)
                if isinstance(word, dict) and 'palabra' in word:
                    if word['palabra'] == 'Te Quiero':
                        icono_base64 = load_image_as_base64('tequiero.png')
                        return jsonify({'palabra': 'Te Quiero', 'icono': icono_base64})
                    elif word['palabra'] == 'B':
                        icono_base64 = load_image_as_base64('B.png')
                        return jsonify({'palabra': 'B', 'icono': icono_base64})
                    elif word['palabra'] == 'D':
                        icono_base64 = load_image_as_base64('D.png')
                        return jsonify({'palabra': 'D', 'icono': icono_base64})
                    elif word['palabra'] == 'C':
                        icono_base64 = load_image_as_base64('C.png')
                        return jsonify({'palabra': 'C', 'icono': icono_base64})
        else:
            return jsonify({'word': 'No se detectaron manos'})
            
    return Response(response='Imagen no válida', status=400)

if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(palabras_api)
    app.run(host='0.0.0.0', port=5000, debug=True)
