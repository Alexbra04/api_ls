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
    thumb_pip = (int(hand_landmarks.landmark[2].x * image_width),
                 int(hand_landmarks.landmark[2].y * image_height))
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


    if thumb_tip[1] < index_finger_tip[1] and thumb_tip[1] < middle_finger_tip[1] and thumb_tip[1] < ring_finger_tip[1] and thumb_tip[1] < pinky_tip[1]:
        letra = 'A'
        icono_base64 = load_image_as_base64('A.png')
        
    elif index_finger_pip[1] - index_finger_tip[1]>0 and pinky_pip[1] - pinky_tip[1] > 0 and \
        middle_finger_pip[1] - middle_finger_tip[1] >0 and ring_finger_pip[1] - ring_finger_tip[1] >0 and \
            middle_finger_tip[1] - ring_finger_tip[1] <0 and abs(thumb_tip[1] - ring_finger_pip2[1])<90:
        letra = 'B'
        icono_base64 = load_image_as_base64('B.png')

    elif (distancia_euclidiana(thumb_tip, middle_finger_tip) < 65 and 
          distancia_euclidiana(thumb_tip, ring_finger_tip) < 65 and 
          pinky_pip[1] - pinky_tip[1] < 0 and 
          index_finger_pip[1] - index_finger_tip[1] > 0):
        letra = 'D'
        icono_base64 = load_image_as_base64('D.png')

    elif abs(index_finger_tip[1] - thumb_tip[1]) < 360 and \
        index_finger_tip[1] - middle_finger_pip[1]<0 and index_finger_tip[1] - middle_finger_tip[1] < 0 and \
            index_finger_tip[1] - index_finger_pip[1] > 0:
        letra = 'C'
        icono_base64 = load_image_as_base64('C.png')
    
    elif index_finger_pip[1] - index_finger_tip[1] < 0 and pinky_pip[1] - pinky_tip[1] < 0 and \
        middle_finger_pip[1] - middle_finger_tip[1] < 0 and ring_finger_pip[1] - ring_finger_tip[1] < 0 \
            and abs(index_finger_tip[1] - thumb_tip[1]) > 50 and \
                thumb_tip[1] - index_finger_tip[1] > 0 \
                and thumb_tip[1] - middle_finger_tip[1] > 0 \
                and thumb_tip[1] - ring_finger_tip[1] > 0 \
                and thumb_tip[1] - pinky_tip[1] > 0:
            letra = 'E'
            icono_base64 = load_image_as_base64('E.png')

    elif  pinky_pip[1] - pinky_tip[1] > 0 and middle_finger_pip[1] - middle_finger_tip[1] > 0 and \
        ring_finger_pip[1] - ring_finger_tip[1] > 0 and index_finger_pip[1] - index_finger_tip[1] < 0 \
            and abs(thumb_pip[1] - thumb_tip[1]) > 0 and distancia_euclidiana(index_finger_tip, thumb_tip) <65:
        letra = 'F'
        icono_base64 = load_image_as_base64('E.png')

    elif index_finger_pip[1] - index_finger_tip[1]>0 and pinky_pip[1] - pinky_tip[1] < 0 and \
        middle_finger_pip[1] - middle_finger_tip[1] >0 and ring_finger_pip[1] - ring_finger_tip[1] <0 and \
            middle_finger_tip[1] - ring_finger_tip[1] <0 and abs(thumb_tip[1] - ring_finger_pip2[1])<20:       
        letra = 'H'
        icono_base64 = load_image_as_base64('H.png')
        
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
    elif distancia_euclidiana(thumb_tip, middle_finger_tip) > 178 \
        and distancia_euclidiana(thumb_tip, ring_finger_tip) > 178 \
        and  pinky_pip[1] - pinky_tip[1]<0\
        and index_finger_pip[1] - index_finger_tip[1]>0:
        letra = 'L'
        icono_base64 = load_image_as_base64('L.png')

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
    
    return {'letra': letra, 'icono': icono_base64}

def rotar_imagen_a_vertical(image):
    # Obtener el alto y el ancho de la imagen
    height, width = image.shape[:2]
    
    # Si la imagen es más ancha que alta, rotar 90 grados hacia la izquierda
    if width > height:
        # Rotar la imagen 90 grados a la izquierda
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


# Ruta para detectar gestos
@abecedario_api.route('/detectar_abecedario', methods=['POST'])
def detectar_abecedario():
    data = request.get_json()
    image_base64 = data.get('image')
    
    if image_base64:

        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        

        image = rotar_imagen_a_vertical(image)


        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_bounding_box(image, hand_landmarks)
                gesture = procesar_gesto(hand_landmarks, image)
                print("Gesto detectado:", gesture)
                if isinstance(gesture, dict) and 'letra' in gesture:
                    if gesture['letra'] == 'A':
                        icono_base64 = load_image_as_base64('A.png')
                        return jsonify({'letra': 'A', 'icono': icono_base64})
                    elif gesture['letra'] == 'B':
                        icono_base64 = load_image_as_base64('B.png')
                        return jsonify({'letra': 'B', 'icono': icono_base64})
                    elif gesture['letra'] == 'D':
                        icono_base64 = load_image_as_base64('D.png')
                        return jsonify({'letra': 'D', 'icono': icono_base64})
                    elif gesture['letra'] == 'C':
                        icono_base64 = load_image_as_base64('C.png')
                        return jsonify({'letra': 'C', 'icono': icono_base64}) 
                    elif gesture['letra'] == 'E':
                        icono_base64 = load_image_as_base64('E.png')
                        return jsonify({'letra': 'E', 'icono': icono_base64})   
                    elif gesture['letra'] == 'F':
                        icono_base64 = load_image_as_base64('F.png')
                        return jsonify({'letra': 'F', 'icono': icono_base64})    
                    elif gesture['letra'] == 'L':
                        icono_base64 = load_image_as_base64('L.png')
                        return jsonify({'letra': 'L', 'icono': icono_base64})   
                    elif gesture['letra'] == 'H':
                        icono_base64 = load_image_as_base64('H.png')
                        return jsonify({'letra': 'H', 'icono': icono_base64})                                                                                                                                                                
        else:
            return jsonify({'gesture': 'No se detectaron manos'})
    
    return Response(response='Imagen no válida', status=400)

if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(abecedario_api)
    app.run(host='0.0.0.0', port=5000, debug=True)
