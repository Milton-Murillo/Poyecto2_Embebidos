# script_1_export_model.py
from ultralytics import YOLO

# Cargar el modelo preentrenado de YOLOv5 (por ejemplo, yolov5s)
model = YOLO("yolov5s.pt")  # Aquí cargamos el modelo preentrenado

# Exportar solo los pesos en formato ONNX
model.export(format="onnx")  # Esto solo exportará los pesos y no el código

# Confirmación de la exportación
print("Modelo exportado como yolov5s.onnx")
