import cv2
import numpy as np
import os
import subprocess

# Configuración esencial para GUI en Linux
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Forzar backend X11

# Configuración de la ventana
MENU_WIDTH = 640
MENU_HEIGHT = 480
WINDOW_NAME = "PongMenu"  # Nombre sin espacios ni caracteres especiales
selected_version = "Pong 2"

# Estados del menú
MAIN_MENU = 0
SETTINGS_MENU = 1
current_menu = MAIN_MENU

# Coordenadas de los botones (mejoradas para mayor precisión)
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

def draw_button(frame, text, position):
    (x1, y1), (x2, y2) = position
    # Rectángulo con borde más grueso
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
    # Texto centrado mejorado
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    text_x = x1 + ((x2 - x1) - text_width) // 2
    text_y = y1 + ((y2 - y1) + text_height) // 2
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

def draw_main_menu(frame):
    frame[:] = (30, 30, 30)  # Fondo gris oscuro
    draw_button(frame, "Jugar", button_positions['main']['play'])
    draw_button(frame, "Ajustes", button_positions['main']['settings'])
    draw_button(frame, "Salir", button_positions['main']['exit'])
    # Título con sombra
    cv2.putText(frame, "Menú Principal", (235, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4)
    cv2.putText(frame, "Menú Principal", (235, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)

def draw_settings_menu(frame):
    frame[:] = (30, 30, 30)
    draw_button(frame, "Pong 2", button_positions['settings']['pong2'])
    draw_button(frame, "Versión Retro", button_positions['settings']['retro'])
    draw_button(frame, "Volver", button_positions['settings']['back'])
    # Indicador de selección
    cv2.putText(frame, f"Versión actual: {selected_version}", 
                (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
    # Título con estilo
    cv2.putText(frame, "Selecciona versión:", 
                (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)

def mouse_callback(event, x, y, flags, param):
    global current_menu, selected_version
    
    if event == cv2.EVENT_LBUTTONDOWN:
        try:
            if current_menu == MAIN_MENU:
                # Detección de clic con márgenes
                if 200 <= x <= 440:
                    if 150 <= y <= 230:
                        subprocess.Popen(["python", "pong2.py" if selected_version == "Pong 2" else "pong_retro.py"])
                        cv2.destroyAllWindows()
                        return
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

# Inicialización de ventana
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(WINDOW_NAME, MENU_WIDTH, MENU_HEIGHT)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

# Bucle principal mejorado
while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
    frame = np.zeros((MENU_HEIGHT, MENU_WIDTH, 3), dtype=np.uint8)
    
    if current_menu == MAIN_MENU:
        draw_main_menu(frame)
    else:
        draw_settings_menu(frame)
    
    cv2.imshow(WINDOW_NAME, frame)
    
    # Manejo de salida con timeout controlado
    key = cv2.waitKey(30)
    if key == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
