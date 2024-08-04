import cv2

# Función para capturar un frame
def capturar_frame():
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        return None

# Función para mostrar un frame (puedes usar esta para depuración)
def mostrar_frame():
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
