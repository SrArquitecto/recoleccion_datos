import time
import threading
import os
import mss
import numpy as np
import cv2
from pynput import keyboard
from datetime import datetime
from deteccion import YoloModelInterface, YoloModel
from segmentacion import MaskGeneratorInterface, BinaryMaskGenerator
from keylogger import KeyLogger, KeyLoggerInterface

class Control():

    def __init__(self):
        self.capturar = False
        self.salir = False
        self.listener_thread = threading.Thread(target=self._run)
        self.listener_thread.start()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.f6:
                self.capturar = not self.capturar
            elif key == keyboard.Key.f12:
                self.salir = True
        except AttributeError:
            pass

    def _on_release(self, key):
        # Implement any functionality needed on key release
        pass

    def guardar_imagen(self, imagen, tstamp, nuevo_ancho=1024, nuevo_alto=576):
        """
        Guarda la imagen después de redimensionarla.
        :param imagen: Imagen a guardar (numpy array).
        :param tstamp: Timestamp para el nombre del archivo.
        :param nuevo_ancho: El nuevo ancho de la imagen después de redimensionarla.
        :param nuevo_alto: El nuevo alto de la imagen después de redimensionarla.
        """
        if not os.path.exists("./train/capturas"):
            os.makedirs("./train/capturas")
        
        # Redimensionar la imagen a las dimensiones deseadas
        imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto))
        
        # Guardar la imagen redimensionada
        cv2.imwrite(f"./train/capturas/{tstamp}.jpg", imagen_redimensionada)

    def iniciar(self, detector: YoloModelInterface, segmentador: MaskGeneratorInterface, keylogger: KeyLoggerInterface):
        start_time = time.time()
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            while True:
                if self.salir:
                    print("adiosssssss!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    break
                if not self.capturar:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 0.1:
                        screenshot = sct.grab(monitor)
                        imagen = np.array(screenshot)
                        imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_BGRA2BGR)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                        print("Saving key logs...")
                        keylogger.save_keys(timestamp)
                        print("Running inference...")
                        detector.inferencia(imagen_bgr)
                        
                        print("Running segmentation...")
                        segmentador.inferencia(imagen_bgr)
                                
                        print("Obtaining detections...")
                        detecciones = detector.obtener_detecciones()
                        print(f"Detections: {detecciones}")
                                
                        print("Generating mask...")
                        segmentador.generar_mascara()
                        print("Saving detections...")
                        detector.guardar_deteciones(timestamp)       
                        self.guardar_imagen(imagen, timestamp)
                        print("Saving mask...")
                        segmentador.guardar_mascara(timestamp)
                
                #sleep(1)

    def _run(self):
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

if __name__ == "__main__":
    control = Control()
    det= YoloModel("./models/det.pt")
    seg = BinaryMaskGenerator("./models/seg.pt")
    key = KeyLogger()
    control.iniciar(det, seg, key)
