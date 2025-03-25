import mss
import time
import datetime
from pynput import keyboard, mouse
import threading
import os

# Variables globales para almacenar las acciones
held_keys = set()         # Almacena las teclas que se mantienen presionadas
mouse_buttons = set()     # Almacena los clics del ratón por ciclo
last_mouse_position = None  # Solo un movimiento del ratón por captura

# Función que registra las teclas presionadas
def on_press(key):
    try:
        held_keys.add(key.char)  # Teclas alfanuméricas
    except AttributeError:
        key_name = str(key).replace("Key.", "")
        held_keys.add(key_name)

# Función que detecta cuando se suelta una tecla
def on_release(key):
    try:
        held_keys.discard(key.char)
    except AttributeError:
        key_name = str(key).replace("Key.", "")
        held_keys.discard(key_name)

# Función que registra los clics del ratón
def on_click(x, y, button, pressed):
    if pressed:
        if button == mouse.Button.left:
            mouse_buttons.add("LMB")
        elif button == mouse.Button.right:
            mouse_buttons.add("RMB")
        elif button == mouse.Button.middle:
            mouse_buttons.add("MMB")

# Función que registra el movimiento del ratón (solo un registro por captura)
def on_move(x, y):
    global last_mouse_position
    last_mouse_position = f"({x}, {y})"

# Iniciar el listener de teclas y de ratón en hilos separados
def start_key_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def start_mouse_listener():
    with mouse.Listener(on_move=on_move, on_click=on_click) as listener:
        listener.join()

# Crear el directorio donde se guardarán las capturas y los archivos de texto
output_dir = "capturas_y_eventos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Función principal que hace las capturas y guarda los archivos
def take_screenshot_and_log():
    global last_mouse_position
    with mss.mss() as sct:
        while True:
            # Obtener la hora actual para usarla en el nombre del archivo
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")

            # Tomar la captura de pantalla
            screenshot = sct.shot(output=os.path.join(output_dir, f"captura_{timestamp}.png"))

            # Formatear los eventos en el nuevo formato solicitado
            keys_str = f"({', '.join(held_keys)})" if held_keys else "()"
            mouse_str = f"({', '.join(mouse_buttons)})" if mouse_buttons else "()"
            mouse_pos_str = last_mouse_position if last_mouse_position else "()"

            # Guardar eventos en una sola línea dentro del .txt
            with open(os.path.join(output_dir, f"captura_{timestamp}.txt"), 'w') as f:
                f.write(f"{keys_str} {mouse_str} {mouse_pos_str}")

            # Limpiar los registros después de guardar
            mouse_buttons.clear()
            last_mouse_position = None  # Restablecer la posición del ratón para la siguiente captura

            # Esperar 0.1 segundos antes de la siguiente captura
            time.sleep(0.1)

# Iniciar el listener de teclas en un hilo
listener_thread_key = threading.Thread(target=start_key_listener)
listener_thread_key.daemon = True
listener_thread_key.start()

# Iniciar el listener de ratón en otro hilo
listener_thread_mouse = threading.Thread(target=start_mouse_listener)
listener_thread_mouse.daemon = True
listener_thread_mouse.start()

# Iniciar el ciclo de captura y registro
take_screenshot_and_log()

