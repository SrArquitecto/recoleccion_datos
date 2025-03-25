from abc import ABC, abstractmethod
from pynput import keyboard, mouse
import threading
import os

class KeyLoggerInterface(ABC):
    @abstractmethod
    def save_keys(self, tstamp):
        pass


class KeyLogger(KeyLoggerInterface):
    def __init__(self, output_dir="./train/keys/"):
        self.held_keys = set()         # Almacena las teclas presionadas
        self.mouse_buttons = set()     # Almacena los clics del ratón
        self.last_mouse_position = None  # Registro único del movimiento del ratón
        self.output_dir = output_dir
        self.teclas_permitidas = {
            '1', '2', '4', 'w', 'a', 's', 'd', 'c', 'g', 'h', 'ctrl', 'shift', 'tab', 'space'
        }
        self._run()

    # Métodos de manejo de eventos
    def _on_press(self, key):
        try:
            key_str = key.char if hasattr(key, 'char') else str(key).replace("Key.", "").lower()
            # Solo agregar la tecla si es permitida
            if key_str in self.teclas_permitidas:
                self.held_keys.add(key_str)
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            key_str = key.char if hasattr(key, 'char') else str(key).replace("Key.", "").lower()
            # Solo remover la tecla si estaba en las teclas permitidas
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

    # Métodos para iniciar los listeners
    def _start_key_listener(self):
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

    def _start_mouse_listener(self):
        with mouse.Listener(on_move=self._on_move, on_click=self._on_click) as listener:
            listener.join()

    # Método para capturar y registrar eventos
    def save_keys(self, tstamp):
        #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        keys_str = f"({', '.join(self.held_keys)})" if self.held_keys else "()"
        mouse_str = f"({', '.join(self.mouse_buttons)})" if self.mouse_buttons else "()"
        mouse_pos_str = self.last_mouse_position if self.last_mouse_position else "()"

        with open(os.path.join(self.output_dir, f"{tstamp}.txt"), 'w') as f:
            f.write(f"{keys_str} {mouse_str} {mouse_pos_str}")

        self.mouse_buttons.clear()
        self.last_mouse_position = None

    # Método para ejecutar todo en hilos separados
    def _run(self):
        threading.Thread(target=self._start_key_listener, daemon=True).start()
        threading.Thread(target=self._start_mouse_listener, daemon=True).start()

if __name__ == "__main__":
    logger = KeyLogger()


