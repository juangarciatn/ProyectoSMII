import sys
import subprocess
import os
import venv
import threading
import concurrent.futures
import time

# --- Variables globales ---
PONG2 = "pong2.py"
PONGRETRO = "pong_retro.py"
GIF_PATH = "ponggif.gif"
WINDOW_NAME = "PongMenu"
CAMERA_RESOLUTION = (640, 480)
DISPLAY_RESOLUTION = (640, 480)
FRAME_SKIP = 2
selected_version = "Pong 2"
debug_enabled = False
rectangle_enabled = False
use_mouse = True  # Modo ratón/hand tracking inicializado por defecto

REQUIRED_PACKAGES = {
    "cv2": "opencv-python==4.9.0.80",
    "numpy": "numpy==1.26.4",
    "mediapipe": "mediapipe==0.10.21",
    "imageio": "imageio==2.34.0",
    "Pillow": "Pillow==10.3.0",
    "pygame": "pygame==2.5.0",
    "screeninfo": "screeninfo==0.8.1"
}

def install_dependencies():
    venv_dir = os.path.join(os.path.dirname(__file__), "pong_venv")
    
    # Configuración correcta de rutas para Windows
    if os.name == 'nt':
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_dir, "bin", "python")
    
    if not os.path.exists(venv_dir):
        print("Creando entorno virtual...")
        venv.create(venv_dir, with_pip=True)
    
    # Instalación con timeout extendido y manejo de errores
    for package in REQUIRED_PACKAGES.values():
        print(f"Instalando {package}...")
        try:
            subprocess.run(
                [python_path, "-m", "pip", "install", "--default-timeout=1000", package],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            print(f"Error instalando {package}: {e}")
            sys.exit(1)
    
    # Verificación final
    print("Verificando dependencias...")
    try:
        subprocess.run(
            [python_path, "-c", "import cv2, numpy, mediapipe, imageio, PIL, pygame, screeninfo"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error verificando dependencias: {e}")
        sys.exit(1)
    
    # Reinicio limpio para Windows
    print("Reiniciando aplicación...")
    os.execl(python_path, python_path, __file__)

try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from PIL import Image, ImageSequence
    import concurrent.futures
    from screeninfo import get_monitors
    import pygame
except ImportError:
    install_dependencies()

# --- Configuración inicial ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3,
    model_complexity=0
)

# Inicializar pygame
pygame.mixer.init(frequency=44100, size=-16, channels=2)
pygame.init()

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
menu_cache = {'main': None, 'settings': None}

# --- Configuración de pantalla ---
try:
    monitor = get_monitors()[0]
    screen_width, screen_height = monitor.width, monitor.height
except:
    screen_width, screen_height = 1920, 1080

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow(WINDOW_NAME, screen_width, screen_height)
cv2.moveWindow(WINDOW_NAME, 0, 0)

# --- Configuración de cámara ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cámara no detectada")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

# --- Carga del GIF ---
def load_optimized_gif():
    try:
        gif = Image.open(GIF_PATH)
        frames = []
        for i, frame in enumerate(ImageSequence.Iterator(gif)):
            if i % 2 == 0:
                frame = frame.convert("RGBA").resize(
                    (screen_width, screen_height),
                    Image.Resampling.NEAREST
                )
                np_frame = np.array(frame)
                cv_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGBA2BGRA)
                frames.append(cv_frame)
        return frames, max(30, gif.info.get('duration', 50))
    except Exception as e:
        print(f"Error GIF: {e}")
        return None, 30

gif_frames, frame_delay = load_optimized_gif()
current_gif_frame = 0

# --- Estados y configuraciones ---
current_menu = 0  # 0: MAIN_MENU, 1: SETTINGS_MENU
cursor_pos = None
click_detected = False
mouse_click = False
hand_results = None

button_layout = {
    'main': {'play': (0.25, 0.2, 0.75, 0.35), 'settings': (0.25, 0.4, 0.75, 0.55), 'exit': (0.25, 0.6, 0.75, 0.75)},
    'settings': {'pong2': (0.25, 0.2, 0.75, 0.3), 'retro': (0.25, 0.35, 0.75, 0.45), 'debug': (0.25, 0.5, 0.75, 0.6), 'rectangles': (0.25, 0.65, 0.75, 0.75), 'back': (0.25, 0.8, 0.75, 0.9)}
}

def load_sound(filename):
    if not os.path.exists(filename):
        return None
    sound = pygame.mixer.Sound(filename)
    sound.set_volume(1.0)
    return sound

click_sound = load_sound("click_sound.mp3")

# --- Callback de ratón ---
def mouse_callback(event, x, y, flags, param):
    global cursor_pos, mouse_click, use_mouse
    use_mouse = True
    cursor_pos = (x, y)
    mouse_click = True if event == cv2.EVENT_LBUTTONDOWN else False

cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

# --- Sistema de caché de menús ---
button_coords = {}
for menu_type in ['main', 'settings']:
    button_coords[menu_type] = {}
    for btn, coords in button_layout[menu_type].items():
        x1 = int(coords[0] * screen_width)
        y1 = int(coords[1] * screen_height)
        x2 = int(coords[2] * screen_width)
        y2 = int(coords[3] * screen_height)
        button_coords[menu_type][btn] = ((x1, y1), (x2, y2))

def build_menu_cache():
    global selected_version, debug_enabled, rectangle_enabled
    for menu_type in ['main', 'settings']:
        overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
        if menu_type == 'main':
            draw_main_menu(overlay)
        else:
            temp_version = selected_version
            temp_debug = debug_enabled
            temp_rect = rectangle_enabled
            draw_settings_menu(overlay)
            selected_version = temp_version
            debug_enabled = temp_debug
            rectangle_enabled = temp_rect
        menu_cache[menu_type] = overlay

# --- Funciones de dibujo ---
def draw_button(frame, text, position, checked=False):
    (x1, y1), (x2, y2) = position
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30, 200), -1)
    border_color = (0, 255, 0) if checked else (255, 255, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)
    text_scale = 0.8 if (x2 - x1) > 500 else 0.5
    thickness = 2 if (x2 - x1) > 500 else 1
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, text_scale, thickness)
    tx = x1 + ((x2 - x1) - text_size[0][0]) // 2
    ty = y1 + ((y2 - y1) + text_size[0][1]) // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, text_scale, (255, 255, 0), thickness)

def draw_main_menu(frame):
    cv2.putText(frame, "PONG AR", (int(screen_width*0.15), int(screen_height*0.1)),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 0), 3)
    for btn in ['play', 'settings', 'exit']:
        draw_button(frame, ['Jugar', 'Ajustes', 'Salir'][['play', 'settings', 'exit'].index(btn)], 
                   button_coords['main'][btn])

def draw_settings_menu(frame):
    states = {'pong2': selected_version == "Pong 2", 'retro': selected_version == "retro",
              'debug': debug_enabled, 'rectangles': rectangle_enabled}
    for btn in ['pong2', 'retro', 'debug', 'rectangles', 'back']:
        text = ['Pong 2', 'Version Retro', 'Modo Debug', 'Modo Rectangulo', 'Volver'][['pong2', 'retro', 'debug', 'rectangles', 'back'].index(btn)]
        draw_button(frame, text, button_coords['settings'][btn], states.get(btn, False))

build_menu_cache()

# --- Detección de manos ---
def async_hand_detection():
    global cursor_pos, click_detected, hand_results
    while True:
        for _ in range(FRAME_SKIP):
            cap.grab()
        
        ret, frame = cap.retrieve()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        future = executor.submit(hands.process, rgb_frame)
        hand_results = future.result()
        
        if hand_results.multi_hand_landmarks:
            hand = hand_results.multi_hand_landmarks[0]
            idx = mp_hands.HandLandmark.INDEX_FINGER_TIP
            thumb = mp_hands.HandLandmark.THUMB_TIP
            
            idx_x = hand.landmark[idx].x
            idx_y = hand.landmark[idx].y
            thumb_x = hand.landmark[thumb].x
            
            cursor_pos = (int(idx_x * screen_width), int(idx_y * screen_height))
            click_detected = np.hypot(idx_x - thumb_x, idx_y - hand.landmark[thumb].y) < 0.05

threading.Thread(target=async_hand_detection, daemon=True).start()

# --- Sistema de sonido ---
def play_click_sound():
    if click_sound:
        channel = pygame.mixer.Channel(1)
        channel.set_volume(1.0, 1.0)
        channel.play(click_sound)

# --- Manejo de interacciones ---
def handle_clicks():
    global current_menu, selected_version, debug_enabled, rectangle_enabled
    x, y = cursor_pos
    menu_type = 'main' if current_menu == 0 else 'settings'
    
    for button_name in button_layout[menu_type]:
        (x1, y1), (x2, y2) = button_coords[menu_type][button_name]
        if x1 <= x <= x2 and y1 <= y <= y2:
            if current_menu == 0:
                if button_name == 'play':
                    venv_dir = os.path.join(os.path.dirname(__file__), "pong_venv")
                    if os.name == 'nt':
                        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
                    else:
                        python_path = os.path.join(venv_dir, "bin", "python")
                    script = PONG2 if selected_version == "Pong 2" else PONGRETRO
                    args = [python_path, script]
                    if debug_enabled: args.append("debug")
                    if rectangle_enabled: args.append("rectangles")
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    subprocess.Popen(args)
                    os._exit(0)
                elif button_name == 'settings':
                    current_menu = 1
                elif button_name == 'exit':
                    cap.release()
                    cv2.destroyAllWindows()
                    os._exit(0)
            else:
                if button_name == 'pong2':
                    selected_version = "Pong 2"
                elif button_name == 'retro':
                    selected_version = "retro"
                elif button_name == 'debug':
                    debug_enabled = not debug_enabled
                elif button_name == 'rectangles':
                    rectangle_enabled = not rectangle_enabled
                elif button_name == 'back':
                    current_menu = 0
            build_menu_cache()
            break

# --- Bucle principal ---
while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
    if gif_frames:
        bg_frame = gif_frames[current_gif_frame].copy()
        current_gif_frame = (current_gif_frame + 1) % len(gif_frames)
    else:
        bg_frame = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    
    current_menu_type = 'main' if current_menu == 0 else 'settings'
    cv2.addWeighted(bg_frame, 1.0, menu_cache[current_menu_type], 1.0, 0, bg_frame)
    
    current_click = mouse_click if use_mouse else click_detected
    current_cursor = cursor_pos if (use_mouse or cursor_pos) else None
    
    if current_cursor:
        color = (0, 255, 0) if use_mouse else (0, 255, 255)
        marker_size = 20 if use_mouse else 30
        cv2.drawMarker(bg_frame, current_cursor, color, 
                      cv2.MARKER_CROSS, marker_size, 2, cv2.LINE_AA)
        
        if debug_enabled:
            debug_text = [
                f"Modo: {'Ratón' if use_mouse else 'Mano'}",
                f"Posición: {current_cursor}",
                f"Detección mano: {bool(hand_results.multi_hand_landmarks) if hand_results else False}",
                f"Click activo: {current_click}"
            ]
            y_offset = 50
            for text in debug_text:
                cv2.putText(bg_frame, text, (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
    
    if current_click:
        handle_clicks()
        play_click_sound()
        mouse_click = False
    
    key = cv2.waitKey(max(1, frame_delay))
    if key == 27:
        break
    elif key == ord('m'):
        use_mouse = not use_mouse
    
    cv2.imshow(WINDOW_NAME, bg_frame)

cap.release()
cv2.destroyAllWindows()