# --- Bloque de instalación mejorado ---
import sys
import subprocess
import os
import venv
import cv2
import numpy as np
from PIL import Image, ImageSequence  # Nuevas dependencias

REQUIRED_PACKAGES = {
    "cv2": "opencv-python==4.9.0.80",
    "numpy": "numpy==1.26.4",
    "mediapipe": "mediapipe==0.10.21",
    "imageio": "imageio==2.34.0",
    "Pillow": "Pillow==10.3.0"
}

def install_dependencies():
    venv_dir = os.path.join(os.path.dirname(__file__), "pong_venv")
    python_path = os.path.join(venv_dir, "bin", "python")
    
    # Crear entorno virtual si no existe
    if not os.path.exists(venv_dir):
        print("Creando entorno virtual...")
        venv.create(venv_dir, with_pip=True)
    
    # Instalar paquetes con versión específica
    for package in REQUIRED_PACKAGES.values():
        print(f"Instalando {package}...")
        subprocess.run(
            [python_path, "-m", "pip", "install", package],
            check=True,
            stdout=subprocess.DEVNULL
        )
    
    # Verificar instalación
    print("Verificando dependencias...")
    subprocess.run([python_path, "-c", "import cv2, numpy, mediapipe, imageio, PIL"], check=True)
    
    os.execv(python_path, [python_path, __file__])

try:
    import cv2, numpy as np, mediapipe
    from PIL import Image, ImageSequence
except ImportError:
    install_dependencies()

# --- Configuración del GIF de fondo ---
GIF_PATH = "ponggif.gif"  # <--- Coloca aquí la ruta ABSOLUTA a tu GIF
MENU_WIDTH = 640
MENU_HEIGHT = 480
WINDOW_NAME = "PongMenu"
selected_version = "Pong 2"

# Estados del menú
MAIN_MENU = 0
SETTINGS_MENU = 1
current_menu = MAIN_MENU

# Coordenadas de los botones
button_positions = {
    'main': {
        'play': ((200, 150), (440, 230)),
        'settings': ((200, 250), (440, 330)),
        'exit': ((200, 350), (440, 430))
    },
    'settings': {
        'pong2': ((200, 150), (440, 230)),
        'retro': ((200, 250), (440, 330)),
        'back': ((200, 350), (440, 430))
    }
}

def load_gif():
    """Carga el GIF y prepara los frames"""
    try:
        gif = Image.open(GIF_PATH)
        frames = []
        for frame in ImageSequence.Iterator(gif):
            frame = frame.convert("RGBA").resize((MENU_WIDTH, MENU_HEIGHT))
            frames.append(cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA))
        return frames, gif.info['duration']
    except Exception as e:
        print(f"Error cargando GIF: {e}")
        return None, 50  # Delay por defecto si hay error

# Cargar GIF al iniciar
gif_frames, frame_delay = load_gif()
current_gif_frame = 0

def draw_button(frame, text, position):
    """Dibuja botones con fondo semitransparente"""
    (x1, y1), (x2, y2) = position
    
    # Fondo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30, 200), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Borde y texto
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255, 255), 3)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    tx = x1 + ((x2 - x1) - tw) // 2
    ty = y1 + ((y2 - y1) + th) // 2
    cv2.putText(frame, text, (tx + 2, ty + 2), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0, 255), 2)
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0, 255), 2)

def draw_main_menu(frame):
    """Dibuja elementos del menú principal"""
    # Título
    cv2.putText(frame, "Menú Principal", (235, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0, 255), 4)
    cv2.putText(frame, "Menú Principal", (235, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0, 255), 2)
    # Botones
    draw_button(frame, "Jugar", button_positions['main']['play'])
    draw_button(frame, "Ajustes", button_positions['main']['settings'])
    draw_button(frame, "Salir", button_positions['main']['exit'])

def draw_settings_menu(frame):
    """Dibuja elementos del menú de ajustes"""
    cv2.putText(frame, "Selecciona versión:", (200, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200, 255), 2)
    cv2.putText(frame, f"Versión actual: {selected_version}",
                (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0, 255), 2)
    draw_button(frame, "Pong 2", button_positions['settings']['pong2'])
    draw_button(frame, "Versión Retro", button_positions['settings']['retro'])
    draw_button(frame, "Volver", button_positions['settings']['back'])

def mouse_callback(event, x, y, flags, param):
    global current_menu, selected_version

    if event == cv2.EVENT_LBUTTONDOWN:
        try:
            if current_menu == MAIN_MENU:
                if 200 <= x <= 440:
                    if 150 <= y <= 230:
                        venv_python = os.path.join(os.path.dirname(__file__), "pong_venv", "bin", "python")
                        subprocess.Popen([venv_python, "pong2.py" if selected_version == "Pong 2" else "pong_retro.py"])
                        cv2.destroyAllWindows()
                    elif 250 <= y <= 330:
                        current_menu = SETTINGS_MENU
                    elif 350 <= y <= 430:
                        cv2.destroyAllWindows()

            elif current_menu == SETTINGS_MENU:
                if 200 <= x <= 440:
                    if 150 <= y <= 230:
                        selected_version = "Pong 2"
                    elif 250 <= y <= 330:
                        selected_version = "retro"
                    elif 350 <= y <= 430:
                        current_menu = MAIN_MENU
        except Exception as e:
            print(f"Error en evento mouse: {e}")

# Configurar ventana en pantalla completa
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

# Bucle principal con animación
while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
    # Actualizar frame del GIF
    if gif_frames:
        bg_frame = gif_frames[current_gif_frame].copy()
        current_gif_frame = (current_gif_frame + 1) % len(gif_frames)
    else:
        bg_frame = np.zeros((MENU_HEIGHT, MENU_WIDTH, 4), dtype=np.uint8)
    
    # Dibujar interfaz
    overlay = np.zeros((MENU_HEIGHT, MENU_WIDTH, 4), dtype=np.uint8)
    
    if current_menu == MAIN_MENU:
        draw_main_menu(overlay)
    else:
        draw_settings_menu(overlay)
    
    # Combinar capas
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        bg_frame[:, :, c] = (1.0 - alpha) * bg_frame[:, :, c] + alpha * overlay[:, :, c]
    
    # Mostrar frame
    cv2.imshow(WINDOW_NAME, bg_frame)

cv2.destroyAllWindows()