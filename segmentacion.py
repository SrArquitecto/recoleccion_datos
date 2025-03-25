import numpy as np
import cv2
import torch
from ultralytics import YOLO
from abc import ABC, abstractmethod
import os

class MaskGeneratorInterface(ABC):
    @abstractmethod
    def inferencia(self, imagen):
        pass
        
    @abstractmethod
    def generar_mascara(self, dibujar=False):
        pass

    @abstractmethod
    def guardar_mascara(self, tstamp):
        pass

    @abstractmethod
    def obtener_mascara(self):
        pass

    @abstractmethod
    def obtener_resultados(self):
        pass

class BinaryMaskGenerator(MaskGeneratorInterface):
    def __init__(self, ruta_modelo, tamanio_imagen=(1080, 1920), output_dir="./train/mascaras/"):
        """
        Inicializa la clase con el tamaño de la imagen de entrada (por defecto 1920x1080) y el modelo YOLO.
        :param tamanio_imagen: (altura, ancho) de la imagen original.
        :param ruta_modelo: Ruta del modelo YOLO.
        """
        self.tamanio_imagen = tamanio_imagen
        self.modelo = YOLO(ruta_modelo)
        self.resultados = None
        self.image = None
        self.image_results = None
        self.mascara_invertida = np.ones((1080, 1920), dtype=np.uint8)
        self.output_dir = output_dir

    def inferencia(self, imagen, conf=0.3, filtro=None):
        """
        Realiza la inferencia usando el modelo YOLO sobre la imagen de entrada.
        :param imagen: Imagen de entrada en formato numpy array.
        :return: Resultados de la segmentación.
        """
        # Realizamos la inferencia con el modelo YOLO
        if filtro is not None:
            self.resultados = self.modelo(imagen, conf=conf, classes=filtro)  
        else:
            self.resultados = self.modelo(imagen, conf=conf)# Esta línea pasa la imagen al modelo y obtiene las predicciones
        #return resultados  # Suponiendo que el modelo devuelve un listado de resultados


    def generar_mascara(self, dibujar=False):
        """
        Genera una imagen binaria a partir de las detecciones de máscaras YOLO.
        :param detecciones: Resultados de segmentación del modelo YOLO.
        :param confianza: Umbral mínimo de confianza para incluir una máscara.
        :return: Imagen binaria en formato numpy array.
        """
        # Acumulador para las máscaras
        mascaras = []

        # Iteramos sobre los resultados de segmentación
        if self.resultados is None:
            raise ValueError("No se han obtenido resultados de la inferencia. Asegúrese de que se ha ejecutado la inferencia correctamente.")
        
        for resultado in self.resultados:
            if hasattr(resultado, 'masks') and resultado.masks is not None:
                masks_data = resultado.masks.data.cpu().numpy()  # Extraer las máscaras como NumPy array
                for mask in masks_data:
                    # Asegurarse de que la máscara esté en formato binario (0 o 1)
                    mask_resized = (mask > 0).astype(np.uint8)
                    mascaras.append(mask_resized)

        # Sumar las máscaras usando np.sum() y luego redimensionar la máscara acumulada
        if mascaras:
            mascara_acumulada = np.sum(mascaras, axis=0)
            mascara_acumulada = np.clip(mascara_acumulada, 0, 1).astype(np.uint8)  # Limitar los valores a 0 o 1
            # Redimensionar la máscara para que coincida con el tamaño de la imagen original
            mascara_acumulada = cv2.resize(mascara_acumulada, (self.tamanio_imagen[1], self.tamanio_imagen[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mascara_acumulada = np.zeros(self.tamanio_imagen, dtype=np.uint8)
        self.mascara_invertida = mascara_acumulada

        if dibujar:
            self._mostrar_mascara()

    def obtener_resultados(self):
        return self.resultados

    def obtener_mascara(self):
        return self.mascara_invertida

    def guardar_mascara(self, tstamp):
        """
        Guarda la imagen binaria en un archivo.
        :param mascara: Imagen binaria generada.
        :param salida: Ruta donde se guardará la imagen.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        nombre_fichero = tstamp + ".jpg"
        salida = os.path.join(self.output_dir, nombre_fichero)

        if self.mascara_invertida is not None:
            cv2.imwrite(salida, self.mascara_invertida * 255)
        else:
            raise ValueError("La máscara invertida no ha sido generada. Asegúrese de que se ha ejecutado 'generar_mascara' correctamente.")
        
    def _mostrar_mascara(self, ):
        
        self.image_results = self.mascara_invertida * 255
        cv2.imshow("Mascara", self.image_results)