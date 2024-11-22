Proyecto Traductor de Lenguaje de Señas Mexicano (LSM)
Este proyecto es un traductor de lenguaje de señas mexicano (LSM) a texto, utilizando modelos de inteligencia artificial y Mediapipe para la detección de landmarks y reconocimiento de gestos en tiempo real.

Requisitos
Asegúrate de tener instalados los siguientes elementos antes de comenzar:

Python 3.8 o superior
Git para clonar el repositorio
Instalación de Git

Crea un entorno virtual (opcional pero recomendado):

python -m venv venv


Activa el entorno virtual:

En Windows:
venv\Scripts\activate

En macOS/Linux:
source venv/bin/activate


Instala las dependencias necesarias:
pip install -r requirements.txt

Ejecución del Proyecto
Configura la cámara: Asegúrate de que tu cámara esté activada y funcionando, ya que este proyecto necesita acceso a la cámara para capturar video en tiempo real.

Ejecuta el archivo principal:

Para iniciar la aplicación de traducción en tiempo real, usa:
python src/main.py


Grabación de señas (opcional):
Si necesitas grabar nuevas señas, ejecuta:
python src/record_signs.py

Este archivo permite capturar nuevas muestras de señas para entrenar y mejorar el modelo de IA.

Estructura del Proyecto
src/main.py: Archivo principal que ejecuta el traductor de LSM en tiempo real.
src/record_signs.py: Archivo para grabar señas y generar datasets adicionales.
requirements.txt: Lista de dependencias necesarias para el proyecto.
Dependencias Principales
OpenCV: Para procesamiento de video en tiempo real.
Mediapipe: Para detección de landmarks de manos y rostros.
TensorFlow: Para el modelo de IA de reconocimiento de señas.
Notas adicionales
Si deseas detener la ejecución en cualquier momento, puedes presionar la tecla "q"