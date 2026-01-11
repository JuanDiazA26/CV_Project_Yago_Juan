# Proyecto Final Visión por Ordenador I

El proyecto final de la asignatura está dividido en tres módulos principales: la calibración de la cámara posteriormente empleada, un sistema de seguridad basado en el reconocimiento de diferentes patrones en una secuencia concreta y el seguimiento de una pelota de baloncesto en su trayectoria a la canasta junto a una predicción de si entra o no.

Para acceder a esta última funcionalidad será necesario introducir una secuencia de formas en el orden correcto.

## 📷 Calibración de cámara

Todo proyecto de visión por ordenador necesita una **calibración de cámara** previa. Este apartado consiste en la realización de dicha calibración junto a la posterior corrección de las imágenes utilizadas. 

## ⚪️ Sistema de seguridad

El **sistema de seguridad** bloquea el acceso al módulo de tracking. Para poder entrar será necesario introducir cuatro formas diferentes (línea horizontal, línea vertical, línea diagonal y círculo) en la secuencia correcta.

## 🏀 Seguimiento de pelota de baloncesto

Una vez dejado atrás el sistema de seguridad se desbloquea el apartado de **tracking**, que consiste en el seguimiento de una pelota de baloncesto en su trayectoria a la canasta. El filtro de Kalman junto a la segmentación por color de la pelota hará posible predecir su trayectoria y comprobar si entra en la canasta.

## 🛠️ Tecnologías utilizadas

* **Lenguaje de programación:** Python
* **Librerías:** OpenCV, Numpy, ImageIO, glob

## 📋 Requisitos

* Entorno virtual voi-lab (Python 3.9.21)

## 🔧 Instalación

Clona el repositorio:
   ```bash
   git clone [https://github.com/usuario/nombre-del-proyecto.git](https://github.com/usuario/nombre-del-proyecto.git)

   (Poner nombre proyecto)