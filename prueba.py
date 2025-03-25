from segmentacion import MaskGeneratorInterface, BinaryMaskGenerator
from deteccion import YoloModelInterface, YoloModel
import cv2
import numpy as np
from multiprocessing import Process
import mss

class ProcesosParalelos:
    def __init__(self):
        """
        Inicializa la clase con dos objetos del modelo de detección y el tamaño de imagen.
        """
        # Creamos dos objetos de la clase BinaryMaskGenerator y YoloModel
        self.generador_det = YoloModel("./models/det.pt")  # Ruta de tu modelo YOLO para detección
        self.generador_seg = BinaryMaskGenerator("./models/seg.pt")  # Ruta de tu modelo YOLO para segmentación

    def proceso_mascara(self, clase_seg: MaskGeneratorInterface, imagen):
        """
        Procesa las máscaras usando el generador de máscaras.
        """
        # Realiza la inferencia
        clase_seg.inferencia(imagen)
        
        # Muestra la máscara generada
        clase_seg.generar_mascara(True)  # Pasar las detecciones al generar la máscara

    def proceso_deteccion(self, clase_det: YoloModelInterface, imagen):
        """
        Procesa las detecciones de objetos usando el modelo YOLO.
        """
        # Realiza la inferencia
        clase_det.inferencia(imagen)
        clase_det.obtener_detecciones(True)

    def inicio(self):
        """
        Método principal para iniciar el proceso de captura y procesamiento en paralelo.
        """
        # Utiliza mss para capturar la pantalla
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Tomamos el monitor principal
            
            while True:
                # Captura la pantalla
                screenshot = sct.grab(monitor)
                
                # Convierte la captura a un numpy array
                imagen = np.array(screenshot)
                
                # Convierte de BGRA a BGR para OpenCV
                imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_BGRA2BGR)
                
                self.proceso_deteccion(self.generador_det, imagen_bgr)
                self.proceso_mascara(self.generador_seg, imagen_bgr)
                
                # Romper el ciclo si presionamos 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Cerrar las ventanas de OpenCV
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Crear la instancia de la clase
    procesos = ProcesosParalelos()
    
    # Iniciar el ciclo principal
    procesos.inicio()
