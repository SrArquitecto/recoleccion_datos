from abc import ABC, abstractmethod
from pynput import keyboard, mouse
import threading
import multiprocessing
import os
import mss
from time import sleep
from datetime import datetime
import numpy as np
import cv2
from deteccion import YoloModelInterface, YoloModel
from segmentacion import MaskGeneratorInterface, BinaryMaskGenerator

class KeyLoggerInterface(ABC):
    @abstractmethod
    def save_keys(self, tstamp, queue):
        pass

class KeyLogger(KeyLoggerInterface):
    def __init__(self, output_dir="./train/keys/"):
        self.held_keys = set()
        self.mouse_buttons = set()
        self.last_mouse_position = None
        self.output_dir = output_dir
        self.teclas_permitidas = {
            '1', '2', '4', 'w', 'a', 's', 'd', 'c', 'g', 'h', 'ctrl', 'shift', 'tab', 'space'
        }
        self._run()

    def _on_press(self, key):
        try:
            key_str = key.char if hasattr(key, 'char') else str(key).replace("Key.", "").lower()
            if key_str in self.teclas_permitidas:
                self.held_keys.add(key_str)
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            key_str = key.char if hasattr(key, 'char') else str(key).replace("Key.", "").lower()
            if key_str in self.teclas_permitidas:
                self.held_keys.discard(key_str)
        except AttributeError:
            pass

    def _on_click(self, x, y, button, pressed):
        if pressed:
            if button == mouse.Button.left:
                self.mouse_buttons.add("LMB")
            elif button == mouse.Button.right:
                self.mouse_buttons.add("RMB")

    def _on_move(self, x, y):
        self.last_mouse_position = f"({x}, {y})"

    def save_keys(self, tstamp, queue):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        keys_str = f"({', '.join(self.held_keys)})" if self.held_keys else "()"
        mouse_str = f"({', '.join(self.mouse_buttons)})" if self.mouse_buttons else "()"
        mouse_pos_str = self.last_mouse_position if self.last_mouse_position else "()"

        # Enviar la información al proceso principal a través de la cola
        queue.put(f"{keys_str} {mouse_str} {mouse_pos_str}")

        # Limpiar el estado después de guardar
        self.mouse_buttons.clear()
        self.last_mouse_position = None

    def _run(self):
        threading.Thread(target=self._start_key_listener, daemon=True).start()
        threading.Thread(target=self._start_mouse_listener, daemon=True).start()

    def _start_key_listener(self):
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

    def _start_mouse_listener(self):
        with mouse.Listener(on_move=self._on_move, on_click=self._on_click) as listener:
            listener.join()

class Control():
    def __init__(self):
        self.capturar = True
        self.salir = False

    def guardar_imagen(self, imagen, tstamp):
        if not os.path.exists("./train/capturas"):
            os.makedirs("./train/capturas")
        cv2.imwrite(f"./train/capturas/{tstamp}.jpg", imagen)

    def guardar_teclas(self, queue, tstamp):
        # Guardar las teclas en el archivo
        if not queue.empty():
            tecla_data = queue.get()
            with open(f"./train/keys/{tstamp}.txt", 'w') as f:
                f.write(tecla_data)

    def iniciar(self, detector: YoloModelInterface, segmentador: MaskGeneratorInterface, keylogger: KeyLoggerInterface):
        # Usamos colas para sincronizar
        captura_queue = multiprocessing.Queue()
        keylog_queue = multiprocessing.Queue()

        def captura_proceso():
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                while not self.salir:
                    if self.capturar:
                        sleep(0.1)
                        screenshot = sct.grab(monitor)
                        imagen = np.array(screenshot)
                        imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_BGRA2BGR)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                        # Guardamos la imagen en la cola
                        captura_queue.put((imagen_bgr, timestamp))
                        keylogger.save_keys(timestamp, keylog_queue)

        def inferencia_proceso():
            while not self.salir:
                if not captura_queue.empty():
                    imagen, timestamp = captura_queue.get()
                    # Realizamos la inferencia y segmentación
                    detector.inferencia(imagen)
                    segmentador.inferencia(imagen)
                    detecciones = detector.obtener_detecciones()
                    segmentador.generar_mascara()
                    detector.guardar_deteciones(timestamp)
                    segmentador.guardar_mascara(timestamp)
                    self.guardar_imagen(imagen, timestamp)
                    # Guardamos las teclas con el mismo timestamp
                    self.guardar_teclas(keylog_queue, timestamp)

        captura_proceso = multiprocessing.Process(target=captura_proceso)
        inferencia_proceso = multiprocessing.Process(target=inferencia_proceso)

        captura_proceso.start()
        inferencia_proceso.start()

        # Esperamos a que los procesos terminen
        captura_proceso.join()
        inferencia_proceso.join()

if __name__ == "__main__":
    control = Control()
    det = YoloModel("./models/det.pt")
    seg = BinaryMaskGenerator("./models/seg.pt")
    key = KeyLogger()
    control.iniciar(det, seg, key)

