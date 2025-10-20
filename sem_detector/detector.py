from ultralytics import YOLO
import cv2

# Cargar el modelo ONNX
model = YOLO("yolov5su.onnx")

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

# Crear una ventana antes de empezar el ciclo
cv2.namedWindow("Detección YOLO11s", cv2.WINDOW_NORMAL)

# Bucle de captura de video
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame")
        break

    # Realizar la predicción con YOLO (detección en el frame)
    results = model(frame)  # Inferir sobre el frame

    # Si hay resultados, se dibujan las cajas
    if len(results) > 0:
        for result in results:  # Iterar si hay más de un resultado
            frame_with_boxes = result.plot()  # Añade las cajas a la imagen
    else:
        frame_with_boxes = frame  # Si no hay detección, solo mostrar el frame original

    # Mostrar el resultado en una ventana (ya existe la ventana)
    cv2.imshow("Detección YOLO11s", frame_with_boxes)

    # Salir del bucle cuando se presiona la tecla 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:  # 27 es el código ASCII para 'Esc'
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

