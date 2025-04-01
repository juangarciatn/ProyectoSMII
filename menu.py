# --- Bloque de instalación mejorado ---
import sys
import subprocess
import os
import venv
import cv2
import numpy as np
from PIL import Image, ImageSequence

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
    
    if not os.path.exists(venv_dir):
        print("Creando entorno virtual...")
        venv.create(venv_dir, with_pip=True)
    
    for package in REQUIRED_PACKAGES.values():
        print(f"Instalando {package}...")
        subprocess.run(
            [python_path, "-m", "pip", "install", package],
            check=True,
            stdout=subprocess.DEVNULL
        )
    
    print("Verificando dependencias...")
    subprocess.run([python_path, "-c", "import cv2, numpy, mediapipe, imageio, PIL"], check=True)
    
    os.execv(python_path, [python_path, __file__])

try:
    import cv2, numpy as np, mediapipe
    from PIL import Image, ImageSequence
except ImportError:
    install_dependencies()

# --- Configuración adaptable ---
GIF_PATH = "ponggif.gif"
WINDOW_NAME = "PongMenu"
selected_version = "Pong 2"

# Configurar ventana en pantalla completa primero
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# Obtener dimensiones reales de la pantalla
screen_width = cv2.getWindowImageRect(WINDOW_NAME)[2]
screen_height = cv2.getWindowImageRect(WINDOW_NAME)[3]

# Validar dimensiones mínimas
screen_width = max(screen_width, 1920)
screen_height = max(screen_height, 1080)

# Estados del menú
MAIN_MENU = 0
SETTINGS_MENU = 1
current_menu = MAIN_MENU

button_layout = {
    'main': {
        'play': (0.25, 0.2, 0.75, 0.35),
        'settings': (0.25, 0.4, 0.75, 0.55),
        'exit': (0.25, 0.6, 0.75, 0.75)
    },
    'settings': {
        'pong2': (0.25, 0.2, 0.75, 0.35),
        'retro': (0.25, 0.4, 0.75, 0.55),
        'back': (0.25, 0.6, 0.75, 0.75)
    }
}

def get_button_coords(layout, menu_type, button_name):
    x1_perc, y1_perc, x2_perc, y2_perc = layout[menu_type][button_name]
    x1 = int(x1_perc * screen_width)
    y1 = int(y1_perc * screen_height)
    x2 = int(x2_perc * screen_width)
    y2 = int(y2_perc * screen_height)
    return ((x1, y1), (x2, y2))

def load_gif():
    try:
        gif = Image.open(GIF_PATH)
        frames = []
        for frame in ImageSequence.Iterator(gif):
            frame = frame.convert("RGBA").resize(
                (screen_width, screen_height),
                Image.Resampling.LANCZOS
            )
            frames.append(cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA))
        return frames, gif.info.get('duration', 50)
    except Exception as e:
        print(f"Error cargando GIF: {e}")
        return None, 50

gif_frames, frame_delay = load_gif()
current_gif_frame = 0

def draw_button(frame, text, position):
    (x1, y1), (x2, y2) = position
    button_width = x2 - x1
    
    font_scale = max(0.5, button_width / 500)
    thickness = max(1, int(button_width / 300))
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30, 200), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255, 255), thickness)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    tx = x1 + ((x2 - x1) - tw) // 2
    ty = y1 + ((y2 - y1) + th) // 2
    cv2.putText(frame, text, (tx + 2, ty + 2), 
                cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0, 255), thickness)
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 0, 255), thickness)

def draw_main_menu(frame):
    title_scale = max(0.8, screen_width / 1600)
    title_thickness = max(1, int(screen_width / 800))
    
    cv2.putText(frame, "", 
                (int(screen_width*0.15), int(screen_height*0.1)),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 0, 0, 255), 
                title_thickness + 2)
    cv2.putText(frame, "", 
                (int(screen_width*0.15), int(screen_height*0.1)),
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX, title_scale, (255, 255, 0, 255), 
                title_thickness)
    
    for text, key in [("Jugar", 'play'), ("Ajustes", 'settings'), ("Salir", 'exit')]:
        draw_button(frame, text, get_button_coords(button_layout, 'main', key))

def draw_settings_menu(frame):
    title_scale = max(0.6, screen_width / 1600)
    cv2.putText(frame, "Selecciona version:", 
                (int(screen_width*0.1), int(screen_height*0.1)), 
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 200, 200, 255), 2)
    
    cv2.putText(frame, f"Version actual: {selected_version}",
                (int(screen_width*0.1), int(screen_height*0.15)), 
                cv2.FONT_HERSHEY_SIMPLEX, title_scale*0.8, (200, 200, 0, 255), 2)
    
    for text, key in [("Pong 2", 'pong2'), ("Version Retro", 'retro'), ("Volver", 'back')]:
        draw_button(frame, text, get_button_coords(button_layout, 'settings', key))

def mouse_callback(event, x, y, flags, param):
    global current_menu, selected_version
    if event == cv2.EVENT_LBUTTONDOWN:
        try:
            menu_type = 'main' if current_menu == MAIN_MENU else 'settings'
            current_buttons = button_layout[menu_type]
            
            for button_name in current_buttons:
                (x1, y1), (x2, y2) = get_button_coords(button_layout, menu_type, button_name)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if current_menu == MAIN_MENU:
                        if button_name == 'play':
                            venv_python = os.path.join(os.path.dirname(__file__), "pong_venv", "bin", "python")
                            subprocess.Popen([venv_python, "pong2.py" if selected_version == "Pong 2" else "pong_retro.py"])
                            cv2.destroyAllWindows()
                        elif button_name == 'settings':
                            current_menu = SETTINGS_MENU
                        elif button_name == 'exit':
                            cv2.destroyAllWindows()
                    else:
                        if button_name == 'pong2':
                            selected_version = "Pong 2"
                        elif button_name == 'retro':
                            selected_version = "retro"
                        elif button_name == 'back':
                            current_menu = MAIN_MENU
                    break
        except Exception as e:
            print(f"Error en evento mouse: {e}")

cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
    if gif_frames:
        bg_frame = gif_frames[current_gif_frame].copy()
        current_gif_frame = (current_gif_frame + 1) % len(gif_frames)
    else:
        bg_frame = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    
    if current_menu == MAIN_MENU:
        draw_main_menu(overlay)
    else:
        draw_settings_menu(overlay)
    
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        bg_frame[:, :, c] = (1.0 - alpha) * bg_frame[:, :, c] + alpha * overlay[:, :, c]
    
    cv2.imshow(WINDOW_NAME, bg_frame)
    
    key = cv2.waitKey(max(1, frame_delay))
    if key == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()