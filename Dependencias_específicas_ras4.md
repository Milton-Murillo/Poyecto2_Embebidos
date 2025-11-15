# Configuración del entorno para modelo de detección en Raspberry Pi 4 (Python 3.11)

## Resumen técnico

Este documento detalla el proceso completo para configurar una **Raspberry Pi 4** con **Python 3.11** y ejecutar un modelo de detección basado en `TensorFlow Lite (tflite-runtime)`.  
La migración desde **Python 3.13** fue necesaria, ya que los paquetes de `tflite-runtime` y `OpenCV` no poseen soporte oficial para versiones superiores a 3.11.

**Objetivo:** lograr que el script `2cam.py` ejecute correctamente el modelo `1.tflite` utilizando dos cámaras V4L2.

---

## Requisitos previos

- Raspberry Pi 4 (64 bits o 32 bits)
- Raspberry Pi OS (Bookworm)
- Conexión a internet
- Cámaras USB o CSI compatibles con V4L2

---

## Instalación de Python 3.11 y configuración del sistema

Actualiza el sistema e instala Python 3.11:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3.11-distutils python3.11-dev
```

Opcionalmente, establece Python 3.11 como versión predeterminada del sistema:

```bash
sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1
sudo update-alternatives --set python3 /usr/local/bin/python3.11

# Verificar
python3 -V
```

Si coexiste Python 3.13, elimínalo para evitar conflictos:

```bash
sudo apt remove -y python3.13 python3.13-venv python3.13-distutils python3.13-minimal
sudo apt autoremove -y
```

---

## Creación del entorno virtual `myenv`

Crea y activa un entorno virtual aislado para el proyecto:

```bash
python3.11 -m venv ~/myenv
source ~/myenv/bin/activate
python -m pip install --upgrade pip wheel
```

---

## Instalación de dependencias del sistema

Instala las dependencias del sistema necesarias:

```bash
sudo apt install -y python3-opencv python3-numpy v4l-utils libopenblas-dev liblapack-dev
```

---

## Instalación de dependencias de Python

Dentro del entorno virtual `myenv`:

```bash
pip install "numpy<2" opencv-python
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime
```

> **Nota:** se utiliza `numpy < 2` para evitar errores con `tflite-runtime 2.14.0`, el cual fue compilado con NumPy 1.x.

---

## Verificación de instalación

Comprueba que las bibliotecas se cargan correctamente:

```bash
python - <<'PY'
import numpy, cv2
from tflite_runtime.interpreter import Interpreter
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
PY
```

Si no se presentan errores, el entorno está listo.

---

## Ejecución del modelo

Activa el entorno virtual y ejecuta el script:

```bash
source ~/myenv/bin/activate
python 2cam.py --model 1.tflite   --cam0 /dev/video0 --cam1 /dev/video2   --cap_w 640 --cap_h 480 --fourcc MJPG   --threads0 2 --threads1 1 --layout h
```

Ajusta `--cam0` y `--cam1` según los dispositivos detectados en tu sistema.

---

## Versiones recomendadas

| Componente      | Versión estable |
|----------------|-----------------|
| Python         | 3.11.x          |
| NumPy          | 1.26.4          |
| tflite-runtime | 2.14.0          |
| OpenCV         | 4.12.0          |
| pip            | 25.3            |

---

## Notas adicionales

- Mantener `numpy < 2.0` mientras se use `tflite-runtime ≤ 2.14`.
- No se recomienda instalar `tensorflow` completo, ya que es muy pesado para la Raspberry Pi 4.
- El entorno `myenv` debe activarse tras cada reinicio:

  ```bash
  source ~/myenv/bin/activate
  ```

- Para verificar las cámaras disponibles:

  ```bash
  v4l2-ctl --list-devices
  ls -l /dev/video*
  ```

---

## Conclusión

Con esta configuración, la Raspberry Pi 4 queda preparada para ejecutar modelos de detección en tiempo real usando **TensorFlow Lite**, garantizando compatibilidad, estabilidad y rendimiento óptimo con Python 3.11 y bibliotecas ajustadas a ARM 64.
