import numpy as np
import cv2
from ultralytics import YOLO
import os
from abc import ABC, abstractmethod

# Definir la interfaz para el modelo YOLO
class YoloModelInterface(ABC):
    @abstractmethod
    def inferencia(self, imagen, conf=0.5, filtro=None):
        """
        Realiza la inferencia sobre una imagen de entrada utilizando el modelo YOLO.
        :param imagen: Imagen de entrada en formato numpy array.
        :return: Resultados de la inferencia.
        """
        pass

    @abstractmethod
    def iniciar_tracker(self, nodo):
        pass
    
    @abstractmethod
    def actualizar_tracker(self, imagen):
        pass

    @abstractmethod
    def obtener_detecciones(self, dibujar=False):
        pass
    
    @abstractmethod
    def obtener_mas_grande(self, dibujar=False):
        pass
        
    @abstractmethod    
    def dibujar_caja_tracker(self, bbox, color=(0, 0, 255), grosor=2):
        pass

    @abstractmethod
    def guardar_deteciones(self, tstamp):
        pass
        
    @abstractmethod
    def obtener_resultados(self):
        pass

# Implementación de la clase que realiza la inferencia y obtiene las detecciones
class YoloModel(YoloModelInterface):

    class_names_to_id = {
        "accion": 0,
        "jugador": 1,
        "nodo": 2
    }
    
    def __init__(self, ruta, output_dir="./train/detecciones/"):
        """
        Inicializa el modelo YOLO para detección de objetos.
        :param ruta: Ruta del modelo YOLO.
        """
        self.modelo = YOLO(ruta)
        self.resultados = None
        self.tracker = cv2.legacy.TrackerMOSSE_create()
        self.detecciones = []
        self.imagen = None
        self.imagen_resultados = None
        self.imagen_tracker = None
        self.bbox = None
        self.visto = None
        self.output_dir = output_dir

    def inferencia(self, imagen, conf=0.5, filtro=None):
        """
        Realiza la inferencia sobre una imagen de entrada utilizando el modelo YOLO.
        :param imagen: Imagen de entrada en formato numpy array.
        :return: Resultados de la inferencia.
        """
        self.imagen = imagen
        self.imagen_resultados = self.imagen.copy()

        if filtro is None:
            self.resultados = self.modelo(imagen, conf=conf)
        else:
            self.resultados = self.modelo(source=imagen, conf=conf, classes=filtro)

    def obtener_resultados(self):
        return self.resultados

    ######LLEVAR A OTRA CLASE
    def iniciar_tracker(self, nodo):
        _, x, y, h, w, _ = nodo
        self.bbox = (x, y, w, h)
        self.tracker.init(self.imagen, self.bbox)
        return self.bbox
    
    def actualizar_tracker(self, imagen):
        self.visto, self.bbox = self.tracker.update(imagen)
        return self.visto, self.bbox


    def obtener_detecciones(self, dibujar=False):
        """
        Devuelve las detecciones filtradas por confianza y por clase (si se especifica).
        :param confianza: Umbral de confianza para filtrar las detecciones.
        :param filtro: (Opcional) Clase específica para filtrar, si es None devuelve todas las clases.
        :return: Detecciones filtradas como un array con formato (idclase, x, y, altura, ancho, confianza).
        """
        if not self.resultados:
            raise ValueError("Se debe realizar la inferencia primero con el método 'infer'.")

        # Extraer los resultados de la inferencia
        self.detecciones = []
        
        for resultado in self.resultados: 
            for box in resultado.boxes:
                conf = box.conf[0].item()
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cls = int(box.cls[0].item())
                self.detecciones.append((cls, x1, y1, x2, y2, conf))
                #print(f"Clase: {cls}, Coordenadas: ({x1}, {y1}, {x2}, {y2}), Confianza: {conf}")
                if dibujar:
                    self._dibujar_caja(cls, x1, y1, x2, y2, conf)
        if dibujar:
            self._mostrar_imagen()       
        return np.array(self.detecciones)
    
    def obtener_mas_grande(self, dibujar=False):
        """
        Devuelve la detección más grande filtrada por confianza y por clase (si se especifica).
        :param confianza: Umbral de confianza para filtrar las detecciones.
        :param filtro: (Opcional) Clase específica para filtrar, si es None devuelve todas las clases.
        :return: Detección más grande como un array con formato (idclase, x, y, altura, ancho, confianza).
        """
        label_mas_grande = None
        confidence_mas_grande = None
        area_maxima = 0
        x, y, xmax, ymax, w, h = 0, 0, 0, 0, 0, 0  # Inicializar variables para la posición y tamaño

        if not self.resultados:
            raise ValueError("Se debe realizar la inferencia primero con el método 'infer'.")

        for resultado in self.resultados: 
            for box in resultado.boxes:
                conf = box.conf[0].item()
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cls = int(box.cls[0].item())
                self.detecciones.append((cls, x1, y1, x2, y2, conf))
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height

                if area > area_maxima:  # Encontrar la detección más grande
                    area_maxima = area
                    label_mas_grande = cls
                    confidence_mas_grande = conf
                    x, y, w, h, xmax, ymax = x1, y1, width, height, x2, y2

        if dibujar:
            self._dibujar_caja(label_mas_grande, x, y, xmax, ymax, confidence_mas_grande)
            self._mostrar_imagen()
        # Si se encuentra una detección más grande, devolverla
        if label_mas_grande is not None:
            return np.array([(label_mas_grande, x, y, h, w, confidence_mas_grande)])

        # Si no se encuentra ninguna detección, retornar un array vacío
        return np.array([])
    
    
    def _dibujar_caja(self, cls, x1, y1, x2, y2, conf, color=(0, 255, 0), grosor=2):
        """
        Dibuja las cajas delimitadoras y etiquetas sobre la imagen.
        :param imagen: Imagen sobre la que se dibujarán las cajas.
        :param color: Color de las cajas y etiquetas.
        :param grosor: Grosor de las cajas.
        :return: Imagen con las cajas dibujadas.
        """
        
        if self.imagen is not None:
            self.imagen_resultados = self.imagen.copy()   
            cv2.rectangle(self.imagen_resultados, (x1, y1), (x2, y2), color, 2)
            class_name = self.modelo.names[cls]
            label_text = f"{class_name} {conf:.2f}"
            self.imagen_resultados = cv2.putText(self.imagen_resultados, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    
    def guardar_deteciones(self, tstamp):
        """
        Guarda las detecciones en un archivo de texto.
        :param tstamp: Timestamp para el nombre del archivo.
        :param salida: Ruta de salida para el archivo de texto.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        nombre_fichero = tstamp + ".txt"
        salida = os.path.join(self.output_dir, nombre_fichero)

        if self.detecciones:
            with open(salida, 'w') as f:
                for deteccion in self.detecciones:
                    cls, x1, y1, x2, y2, _ = deteccion
                    x1, y1, x2, y2 = self._normalizar_coordenadas(x1, y1, x2, y2)
                    d = f"{cls} {x1} {y1} {x2} {y2}"
                    f.write(f"{d}\n")
        else:
            with open(salida, 'w') as f:
                f.write(" ")
    
    def _normalizar_coordenadas(self, x1, y1, x2, y2, image_width=1920, image_height=1080):
        """
        Normaliza las coordenadas de una caja delimitadora a un rango [0, 1] 
        basado en las dimensiones de la imagen de entrada (1920x1080 por defecto).
        
        :param x1: Coordenada x1 (esquina superior izquierda).
        :param y1: Coordenada y1 (esquina superior izquierda).
        :param x2: Coordenada x2 (esquina inferior derecha).
        :param y2: Coordenada y2 (esquina inferior derecha).
        :param image_width: Ancho de la imagen (por defecto 1920).
        :param image_height: Alto de la imagen (por defecto 1080).
        
        :return: Las coordenadas normalizadas (x1, y1, x2, y2).
        """
        x1_normalized = x1 / image_width
        y1_normalized = y1 / image_height
        x2_normalized = x2 / image_width
        y2_normalized = y2 / image_height

        return x1_normalized, y1_normalized, x2_normalized, y2_normalized

    def _desnormalizar_coordenadas(self, x1, y1, x2, y2, image_width=1920, image_height=1080):
        """
        Desnormaliza las coordenadas de una caja delimitadora a un rango [0, 1]"""
        x1_desnormalizado = x1 * image_width
        y1_desnormalizado = y1 * image_height
        x2_desnormalizado = x2 * image_width
        y2_desnormalizado = y2 * image_height

        return x1_desnormalizado, y1_desnormalizado, x2_desnormalizado, y2_desnormalizado

    def dibujar_caja_tracker(self, bbox, color=(0, 0, 255), grosor=2):
        """
        Dibuja las cajas delimitadoras y etiquetas sobre la imagen.
        :param imagen: Imagen sobre la que se dibujarán las cajas.
        :param color: Color de las cajas y etiquetas.
        :param grosor: Grosor de las cajas.
        :return: Imagen con las cajas dibujadas.
        """
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height

            # Dibuja la caja delimitadora
        imagen = cv2.rectangle(imagen, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, grosor)
        
        return imagen

    def _mostrar_imagen(self):
        """
        Muestra la imagen con las detecciones.
        """
        
        if self.imagen_resultados is not None:
            cv2.imshow("Detecciones", self.imagen_resultados)
