import sys
import subprocess
import os
import venv
import threading
import concurrent.futures
import time
import argparse

# Configurar argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--music-volume", type=float, default=0.5)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--rectangles", action="store_true")
args = parser.parse_args()

# --- Variables globales ---
PONG2 = "pong2.py"
PONGRETRO = "pong_retro.py"
GIF_PATH = "ponggif.gif"
WINDOW_NAME = "PongMenu"
CAMERA_RESOLUTION = (1280, 720)
DISPLAY_RESOLUTION = (640, 480)
FRAME_SKIP = 1
music_volume = args.music_volume
debug_enabled = args.debug
rectangle_enabled = args.rectangles
dragging_volume = False
selected_version = "Pong 2"
use_mouse = True
last_click_time = 0
click_active = False
last_mouse_click = False
click_processed = False


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
    python_path = os.path.join(venv_dir, "Scripts", "python.exe") if os.name == 'nt' else os.path.join(venv_dir, "bin", "python")
    
    # Verificación ultra rápida de entorno válido
    if os.path.exists(venv_dir) and os.path.exists(python_path):
        # Verificar solo un paquete crítico para ahorrar tiempo
        check_cmd = f"import {list(REQUIRED_PACKAGES.keys())[0]}"
        result = subprocess.run([python_path, "-c", check_cmd], capture_output=True)
        if result.returncode == 0:
            print("Entorno virtual ya configurado. Saltando instalación...")
            os.execl(python_path, python_path, os.path.abspath(__file__))
            return

    # Si falla la verificación rápida, instalar normalmente
    if os.path.exists(venv_dir):
        import shutil
        shutil.rmtree(venv_dir)
    
    print("Creando entorno virtual...")
    venv.create(venv_dir, with_pip=True)
    
    pip_path = os.path.join(venv_dir, "Scripts", "pip.exe") if os.name == 'nt' else os.path.join(venv_dir, "bin", "pip")
    
    # Instalación paralelizada
    install_cmd = [pip_path, "install", "--disable-pip-version-check", "--no-input"] + list(REQUIRED_PACKAGES.values())
    
    with subprocess.Popen(install_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        while True:
            output = proc.stdout.readline()
            if output == '' and proc.poll() is not None:
                break
            if output:
                print(output.strip())
    
    print("Reiniciando...")
    os.execl(python_path, python_path, os.path.abspath(__file__))

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
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7,
    model_complexity=0
)


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
        gif = Image.open("assets/"+GIF_PATH)
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
    'main': {
        'play': (0.25, 0.2, 0.75, 0.35),
        'settings': (0.25, 0.4, 0.75, 0.55),
        'exit': (0.25, 0.6, 0.75, 0.75)
    },
    'settings': {
        'pong2': (0.25, 0.1, 0.75, 0.2),
        'retro': (0.25, 0.25, 0.75, 0.35),
        'volume': (0.25, 0.4, 0.75, 0.5),
        'debug': (0.25, 0.55, 0.75, 0.65),
        'rectangles': (0.25, 0.7, 0.75, 0.8),
        'back': (0.25, 0.85, 0.75, 0.95)
    }
}

def load_sound(filename):
    asset_path = os.path.join("assets", filename)
    if not os.path.exists(asset_path):
        return None
    sound = pygame.mixer.Sound(asset_path)
    sound.set_volume(1.0)
    return sound

# Añadir esta función para cargar la música
def setup_background_music():
    music_path = os.path.join("assets", "arcade_acadia.mp3")
    if os.path.exists(music_path):
        pygame.mixer.music.load(music_path)
        pygame.mixer.music.set_volume(music_volume)
        pygame.mixer.music.play(-1)

# Dibuja el menú principal con efecto hover
def draw_main_menu(frame, hovered_button):
    # Título del juego
    cv2.putText(
        frame,
        "PONG AR",
        (int(screen_width * 0.15), int(screen_height * 0.1)),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        (255, 255, 0),
        3
    )
    # Botones del menú
    labels = ['Jugar', 'Ajustes', 'Salir']
    keys = ['play', 'settings', 'exit']
    for key, text in zip(keys, labels):
        is_hovered = (hovered_button == key)
        # No hay estado "checked" en el menú principal
        draw_button(
            frame,
            text,
            button_coords['main'][key],
            checked=False,
            hovered=is_hovered
        )

# Dibuja el menú de ajustes con estados y hover
def draw_settings_menu(frame):
    global music_volume
    # Estados "checked" para cada opción
    states = {
        'pong2': (selected_version == "Pong 2"),
        'retro': (selected_version == "retro"),
        'debug': debug_enabled,
        'rectangles': rectangle_enabled
    }
    # Determinar qué botón está hovered
    hovered_button = get_hovered_button('settings', cursor_pos)
    # Botones de opciones
    option_keys = ['pong2', 'retro', 'debug', 'rectangles', 'back']
    option_labels = ['Pong 2', 'Version Retro', 'Modo Debug', 'Modo Rectangulo', 'Volver']
    for key, text in zip(option_keys, option_labels):
        draw_button(
            frame,
            text,
            button_coords['settings'][key],
            checked=states.get(key, False),
            hovered=(hovered_button == key)
        )
    # Slider de volumen
    x1, y1 = button_coords['settings']['volume'][0]
    x2, y2 = button_coords['settings']['volume'][1]
    # Fondo del slider
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30, 200), -1)
    # Relleno según volumen
    thumb_x = x1 + int(music_volume * (x2 - x1))
    cv2.rectangle(frame, (x1, y1), (thumb_x, y2), (0, 200, 0, 200), -1)
    # Texto de porcentaje
    cv2.putText(
        frame,
        f"Volumen: {int(music_volume * 100)}%",
        (x1 + 10, y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

# Construye y almacena en caché los overlays para los menús principal y ajustes
def build_menu_cache():
    global selected_version, debug_enabled, rectangle_enabled
    for menu_type in ['main', 'settings']:
        # Crear un overlay transparente del tamaño de la pantalla
        overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
        if menu_type == 'main':
            # Dibujar menú principal sin estado hover (se aplica en tiempo real)
            draw_main_menu(overlay, None)
        else:
            # Para el menú de ajustes, preservamos temporalmente los estados modificados durante el dibujo
            temp_version = selected_version
            temp_debug = debug_enabled
            temp_rect = rectangle_enabled
            draw_settings_menu(overlay)
            # Restaurar estados originales para no alterar el flujo de la caché
            selected_version = temp_version
            debug_enabled = temp_debug
            rectangle_enabled = temp_rect
        # Guardar overlay en el diccionario de caché
        menu_cache[menu_type] = overlay

# --- Manejo de interacciones ---
def handle_clicks():
    global current_menu, selected_version, debug_enabled, rectangle_enabled, music_volume, click_processed, dragging_volume
    
    x, y = cursor_pos
    menu_type = 'main' if current_menu == 0 else 'settings'

    # Manejar slider de volumen (prioridad máxima)
    if dragging_volume and current_menu == 1:
        x, y = cursor_pos
        (x1, y1), (x2, y2) = button_coords['settings']['volume']
        if x1 <= x <= x2 and y1 <= y <= y2:
            slider_width = x2 - x1
            new_volume = (x - x1) / slider_width
            music_volume = max(0.0, min(1.0, new_volume))
            pygame.mixer.music.set_volume(music_volume)
            build_menu_cache()
            return  # Salir inmediatamente después de actualizar

    # Procesar otros botones
    for button_name in button_layout[menu_type]:
        (x1, y1), (x2, y2) = button_coords[menu_type][button_name]
        if x1 <= x <= x2 and y1 <= y <= y2:
            if current_menu == 0:
                # Lógica del menú principal
                if button_name == 'play':
                    venv_dir = os.path.join(os.path.dirname(__file__), "pong_venv")
                    python_path = os.path.join(venv_dir, "Scripts", "python.exe") if os.name == 'nt' else os.path.join(venv_dir, "bin", "python")
                    script = PONG2 if selected_version == "Pong 2" else PONGRETRO
                    args = [python_path, script, "--music-volume", str(music_volume)]
                    if debug_enabled: args.append("--debug")
                    if rectangle_enabled: args.append("--rectangles")
                    
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
                    
            else:  # Menú de ajustes
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
            break  # Procesar solo un botón por clic
    
    # Marcar como procesado excepto para arrastre de volumen
    if not dragging_volume:
        click_processed = True

# --- Callback de ratón ---
def mouse_callback(event, x, y, flags, param):
    global cursor_pos, mouse_click, use_mouse, dragging_volume, click_processed, last_mouse_click
    use_mouse = True
    cursor_pos = (x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if not last_mouse_click:
            mouse_click = True
            click_processed = False  # Reiniciar estado de procesado
            # Verificar clic en slider solo si estamos en menú de ajustes
            if current_menu == 1:
                (x1, y1), (x2, y2) = button_coords['settings']['volume']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    dragging_volume = True
            last_mouse_click = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_click = False
        last_mouse_click = False
        click_processed = False  # Permitir nuevo clic
        dragging_volume = False
    elif event == cv2.EVENT_MOUSEMOVE and dragging_volume:
        # Actualizar volumen durante el arrastre (ignorar debounce)
        handle_clicks()

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

# Devuelve el botón sobre el que está el cursor, o None si no hay ninguno
def get_hovered_button(menu_type, cursor_pos):
    if not cursor_pos:
        return None
    x, y = cursor_pos
    # Recorremos cada botón definido en button_layout para el menú dado
    for btn in button_layout[menu_type]:
        (x1, y1), (x2, y2) = button_coords[menu_type][btn]
        # Si el cursor está dentro de las coordenadas del botón, devolvemos su identificador
        if x1 <= x <= x2 and y1 <= y <= y2:
            return btn
    return None

# Dibuja un botón con estado normal, checked y hovered
def draw_button(frame, text, position, checked=False, hovered=False):
    (x1, y1), (x2, y2) = position
    # Fondo semitransparente diferente si está hovered
    bg_color = (60, 60, 60, 200) if hovered else (30, 30, 30, 200)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)

    # Color de borde: verde si está marcado, blanco en estado normal
    border_color = (0, 255, 0) if checked else (255, 255, 255)
    # Si está hovered, usar cyan para resaltar
    if hovered:
        border_color = (0, 200, 200)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

    # Ajuste de escala y grosor de texto según tamaño del botón
    text_scale = 0.8 if (x2 - x1) > 500 else 0.5
    thickness = 2 if (x2 - x1) > 500 else 1
    # Medir tamaño del texto para centrarlo
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, text_scale, thickness)[0]
    tx = x1 + ((x2 - x1) - text_size[0]) // 2
    ty = y1 + ((y2 - y1) + text_size[1]) // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, text_scale, (255, 255, 0), thickness)

build_menu_cache()

# --- Detección de manos ---
def async_hand_detection():
    global cursor_pos, click_detected, hand_results, click_active, last_click_time, dragging_volume, click_processed
    prev_pos = None
    SMOOTHING_FACTOR = 0.15
    CLICK_THRESHOLD = 0.07
    DEBOUNCE_TIME = 0.3  # 300ms entre clics
    MIN_HOLD_TIME = 0.1  # 100ms de contacto mínimo
    contact_start_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        
        if hand_results.multi_hand_landmarks:
            hand = hand_results.multi_hand_landmarks[0]
            idx = mp_hands.HandLandmark.INDEX_FINGER_TIP
            thumb = mp_hands.HandLandmark.THUMB_TIP
            
            # Coordenadas normalizadas
            idx_x = hand.landmark[idx].x
            idx_y = hand.landmark[idx].y
            thumb_x = hand.landmark[thumb].x
            thumb_y = hand.landmark[thumb].y
            
            # Suavizado de posición
            raw_pos = (int(idx_x * screen_width), int(idx_y * screen_height))
            if prev_pos:
                cursor_pos = (
                    int(raw_pos[0] * (1 - SMOOTHING_FACTOR) + prev_pos[0] * SMOOTHING_FACTOR),
                    int(raw_pos[1] * (1 - SMOOTHING_FACTOR) + prev_pos[1] * SMOOTHING_FACTOR)
                )
            else:
                cursor_pos = raw_pos
            prev_pos = cursor_pos
            
            # Cálculo de distancia
            distance = np.hypot(idx_x - thumb_x, idx_y - thumb_y)
            current_time = time.time()
            
            # Lógica de estados mejorada
            if distance < CLICK_THRESHOLD:
                if not click_active:
                    click_active = True
                    contact_start_time = current_time
                    click_processed = False  # Nuevo contacto, permitir procesar
                else:
                    # Verificar tiempo de contacto y debounce
                    if (current_time - contact_start_time >= MIN_HOLD_TIME) and \
                       (current_time - last_click_time >= DEBOUNCE_TIME) and \
                       not click_processed:
                        click_detected = True
                        last_click_time = current_time
                        click_processed = True  # Marcar como procesado
            else:
                if click_active:
                    click_processed = True  # Forzar liberación completa
                click_active = False
                click_detected = False
            
            # Detectar arrastre
            if click_active and (current_time - contact_start_time >= MIN_HOLD_TIME):
                (x1, y1), (x2, y2) = button_coords['settings']['volume']
                x, y = cursor_pos
                if current_menu == 1 and x1 <= x <= x2 and y1 <= y <= y2:
                    dragging_volume = True
                else:
                    dragging_volume = False
            else:
                dragging_volume = False
                
        else:
            click_active = False
            click_detected = False
            dragging_volume = False
            prev_pos = None
            click_processed = True  # Bloquear hasta nuevo contacto

threading.Thread(target=async_hand_detection, daemon=True).start()

# --- Sistema de sonido ---
def play_click_sound():
    if click_sound:
        channel = pygame.mixer.Channel(1)
        channel.set_volume(1.0, 1.0)
        channel.play(click_sound)

try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    print("pygame.mixer inicializado correctamente")
except Exception as e:
    print(f"Error inicializando pygame.mixer: {e}")

setup_background_music()

click_sound = load_sound("click_sound.mp3")

# --- Bucle principal ---
while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
    # Actualizar frame de fondo
    if gif_frames:
        bg_frame = gif_frames[current_gif_frame].copy()
        current_gif_frame = (current_gif_frame + 1) % len(gif_frames)
    else:
        bg_frame = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)

    # Determinar el tipo de menú actual y el botón bajo el cursor
    current_menu_type = 'main' if current_menu == 0 else 'settings'
    hovered_button = get_hovered_button(current_menu_type, cursor_pos)

    # Redibujar el menú en cada frame con el hover actual
    if current_menu == 0:
        menu_overlay = menu_cache['main'].copy()
        draw_main_menu(menu_overlay, hovered_button)
    else:
        menu_overlay = menu_cache['settings'].copy()
        draw_settings_menu(menu_overlay)

    # Combinar fondo y menú
    cv2.addWeighted(bg_frame, 1.0, menu_overlay, 1.0, 0, bg_frame)

    # Determinar modo de control y estado de click
    if hand_results and hand_results.multi_hand_landmarks:
        use_mouse = False
        current_click = click_detected
    else:
        use_mouse = True
        current_click = mouse_click

    # Dibujar cursor y debug si está activado
    if cursor_pos:
        cursor_color = (0, 255, 0) if use_mouse else (0, 255, 255)
        marker_size = 20 if use_mouse else 35
        cv2.drawMarker(bg_frame, cursor_pos, cursor_color, cv2.MARKER_CROSS, marker_size, 2, cv2.LINE_AA)

        if debug_enabled:
            debug_info = [
                f"Modo: {'Raton' if use_mouse else 'Mano'}",
                f"Posicion: {cursor_pos}",
                f"Click detectado: {click_detected}",
                f"Contacto activo: {click_active}",
                f"Ultimo clic: {time.time() - last_click_time:.2f}s",
                f"Arrastrando: {dragging_volume}"
            ]
            y_offset = 50
            for line in debug_info:
                cv2.putText(bg_frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

    # Manejar interacciones de click
    if current_click:
        handle_clicks()
        play_click_sound()
        # Resetear estado de click
        if use_mouse:
            mouse_click = False
        else:
            click_detected = False

    # Actualizar volumen mientras se arrastra el slider
    if dragging_volume:
        handle_clicks()

    # Manejo de teclado
    key = cv2.waitKey(max(1, frame_delay))
    if key == 27:  # ESC
        break
    elif key == ord('m'):
        use_mouse = not use_mouse
    elif key == ord('+') and current_menu == 1:
        music_volume = min(1.0, music_volume + 0.05)
        pygame.mixer.music.set_volume(music_volume)
        build_menu_cache()
    elif key == ord('-') and current_menu == 1:
        music_volume = max(0.0, music_volume - 0.05)
        pygame.mixer.music.set_volume(music_volume)
        build_menu_cache()

    # Mostrar el frame resultante
    cv2.imshow(WINDOW_NAME, bg_frame)

# Limpieza final
cap.release()
cv2.destroyAllWindows()