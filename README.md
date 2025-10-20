# Detector de Objetos con YOLOv5 en Raspberry Pi OS

## 🧩 1. Dependencias necesarias

### 🔹 Librerías de Python
```bash
sudo apt update
sudo apt install python3-pip python3-opencv python3-venv -y
pip3 install ultralytics onnx onnxruntime
```

**Explicación:**
- **ultralytics** → Framework que permite cargar y usar modelos YOLOv5, YOLOv8 y ONNX.  
- **onnx** → Define el formato estándar en que se exporta el modelo.  
- **onnxruntime** → Motor de ejecución optimizado para correr modelos ONNX con buena velocidad.  
- **opencv-python** → Permite acceder a la cámara, capturar video e interactuar visualmente con los resultados.

---

## 📸 2. Dependencias del sistema (Raspberry Pi OS)

```bash
sudo apt install libatlas-base-dev libopenblas-dev libhdf5-dev
sudo apt install libopencv-dev v4l-utils -y
```

Para verificar la cámara:
```bash
v4l2-ctl --list-devices
```

Debe aparecer listada como `/dev/video0` o similar.

---

## 🧠 3. Archivos del proyecto

### a. Modelo exportado
Archivo necesario:
```
yolov5su.onnx
```
Contiene solo los **pesos del modelo YOLOv5**, en formato ONNX.

### b. Script principal
```python
from ultralytics import YOLO
import cv2

# Cargar el modelo ONNX
model = YOLO("yolov5su.onnx")

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Verificar la cámara
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

cv2.namedWindow("Detección YOLOv5", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame")
        break

    results = model(frame)
    if len(results) > 0:
        for result in results:
            frame_with_boxes = result.plot()
    else:
        frame_with_boxes = frame

    cv2.imshow("Detección YOLOv5", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Este script usa el modelo exportado y realiza detección en tiempo real.

---

## ⚙️ 4. Flujo estructurado del sistema

| Etapa | Descripción |
|-------|--------------|
| **1. Captura de imagen** | OpenCV obtiene cada frame desde la cámara. |
| **2. Inferencia con YOLOv5 (ONNX)** | El modelo `yolov5su.onnx` se ejecuta mediante Ultralytics. |
| **3. Detección** | El modelo devuelve las cajas y etiquetas detectadas. |
| **4. Visualización** | OpenCV dibuja las cajas en la ventana. |
| **5. Control del flujo** | Se ejecuta hasta que se presione la tecla `Esc`. |

---

## 🔗 5. Enlaces útiles

- [Documentación Ultralytics](https://docs.ultralytics.com)
- [Modelos YOLOv5 preentrenados](https://github.com/ultralytics/yolov5/releases)
- [ONNX (Open Neural Network Exchange)](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)

---

## 🧾 6. Recomendaciones de rendimiento

- Reducir resolución:
  ```python
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  ```
- Cerrar procesos innecesarios antes de ejecutar.
- Para mayor velocidad, se puede convertir a **TensorRT** o **OpenVINO**, aunque ONNX es suficiente.

---

📘 **Autor:** Josué Marín  
📅 **Proyecto:** Detector de Objetos en Raspberry Pi – YOLOv5 (ONNX)  
🏷️ **Versión:** 1.0
