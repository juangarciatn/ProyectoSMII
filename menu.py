import cv2
import os
import subprocess

# Configuración de la ventana
MENU_WIDTH = 640
MENU_HEIGHT = 480
selected_version = "Pong 2"  # Valor por defecto

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
        'Pong 2': ((200, 150), (440, 230)),
        'retro': ((200, 250), (440, 330)),
        'back': ((200, 350), (440, 430))
    }
}

def draw_button(frame, text, position):
    (x1, y1), (x2, y2) = position
    # Dibujar rectángulo del botón
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # Centrar texto
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_main_menu(frame):
    frame.fill((0, 0, 0))
    draw_button(frame, "Jugar", button_positions['main']['play'])
    draw_button(frame, "Ajustes", button_positions['main']['settings'])
    draw_button(frame, "Salir", button_positions['main']['exit'])
    cv2.putText(frame, "Menú Principal", (240, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_settings_menu(frame):
    frame.fill((0, 0, 0))
    draw_button(frame, "Pong 2", button_positions['settings']['Pong 2'])
    draw_button(frame, "Retro", button_positions['settings']['retro'])
    draw_button(frame, "Volver", button_positions['settings']['back'])
    
    # Mostrar selección actual
    cv2.putText(frame, f"Versión actual: {selected_version}", 
                (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Selecciona versión:", 
                (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def mouse_callback(event, x, y, flags, param):
    global current_menu, selected_version
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_menu == MAIN_MENU:
            # Detectar clic en botones del menú principal
            if button_positions['main']['play'][0][0] < x < button_positions['main']['play'][1][0] and \
               button_positions['main']['play'][0][1] < y < button_positions['main']['play'][1][1]:
                # Lanzar el juego correspondiente
                game_file = "pong2.py" if selected_version == "Pong 2" else "pong_retro.py"
                subprocess.Popen(["python", game_file])
                cv2.destroyAllWindows()
                
            elif button_positions['main']['settings'][0][0] < x < button_positions['main']['settings'][1][0] and \
                 button_positions['main']['settings'][0][1] < y < button_positions['main']['settings'][1][1]:
                current_menu = SETTINGS_MENU
                
            elif button_positions['main']['exit'][0][0] < x < button_positions['main']['exit'][1][0] and \
                 button_positions['main']['exit'][0][1] < y < button_positions['main']['exit'][1][1]:
                cv2.destroyAllWindows()
                
        elif current_menu == SETTINGS_MENU:
            # Detectar clic en botones de ajustes
            if button_positions['settings']['Pong 2'][0][0] < x < button_positions['settings']['Pong 2'][1][0] and \
               button_positions['settings']['Pong 2'][0][1] < y < button_positions['settings']['Pong 2'][1][1]:
                selected_version = "Pong 2"
                
            elif button_positions['settings']['retro'][0][0] < x < button_positions['settings']['retro'][1][0] and \
                 button_positions['settings']['retro'][0][1] < y < button_positions['settings']['retro'][1][1]:
                selected_version = "retro"
                
            elif button_positions['settings']['back'][0][0] < x < button_positions['settings']['back'][1][0] and \
                 button_positions['settings']['back'][0][1] < y < button_positions['settings']['back'][1][1]:
                current_menu = MAIN_MENU

# Crear ventana
cv2.namedWindow("Menú Pong AR")
cv2.setMouseCallback("Menú Pong AR", mouse_callback)

while True:
    frame = np.zeros((MENU_HEIGHT, MENU_WIDTH, 3), dtype=np.uint8)
    
    if current_menu == MAIN_MENU:
        draw_main_menu(frame)
    elif current_menu == SETTINGS_MENU:
        draw_settings_menu(frame)
    
    cv2.imshow("Menú Pong AR", frame)
    
    # Salir con ESC
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()