import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
import subprocess
import random
import sys

# Inicializar pygame
pygame.mixer.init(frequency=44100, size=-16, channels=2)
pygame.init()

# Constantes del juego
COUNT_DOWN_LEFT_HAND = 2
COUNT_DOWN_RIGHT_HAND = 2
RECTANGULO_WIDTH = 30
RECTANGULO_HEIGHT = 90
BALL_SIZE = 10
MIN_SPEED = 5
MAX_SPEED = 40

# Variables de pantalla
WIDTH = 1280
HEIGHT = 720
POS_HORIZONTAL_IZQUIERDA = 0
POS_HORIZONTAL_DERECHA = 0

# Estado del juego
paused = False

# Variables para modos
DEBUG_MODE = False
RECTANGLE_MODE = False

def close():
    print("Closing game...")
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit(0)

def scape_to_menu():
    print("Leaving pong_retro.py...")
    print("Going to menu.py...")
    cv2.destroyAllWindows()
    subprocess.Popen(["python", "menu.py"])

def load_sound(filename):
    if not os.path.exists(filename):
        return None
    sound = pygame.mixer.Sound(filename)
    sound.set_volume(1.0)
    return sound

def reset_ball(direction):
    global ballPosition, ballSpeedX, ballSpeedY, last_touched
    
    ballPosition = [WIDTH // 2, HEIGHT // 2]
    ballSpeedX = MIN_SPEED * direction
    ballSpeedY = MIN_SPEED * random.choice([-1, 1])
    last_touched = None
    time.sleep(1)

def initialize_game():
    global pong_sound, win_sound, countdown_sound
    global left_paddle_y, right_paddle_y
    global ballPosition, ballSpeedX, ballSpeedY
    global left_score, right_score, last_touched
    global hands, mp_hands, mp_drawing
    global WIDTH, HEIGHT, POS_HORIZONTAL_IZQUIERDA, POS_HORIZONTAL_DERECHA
    global RECTANGULO_HEIGHT, DEBUG_MODE, RECTANGLE_MODE
    
    # Verificar argumentos de línea de comandos
    args = [arg.lower() for arg in sys.argv[1:]]
    DEBUG_MODE = "debug" in args
    RECTANGLE_MODE = "rectangle" in args
    
    try:
        screen_info = pygame.display.Info()
        WIDTH = screen_info.current_w
        HEIGHT = screen_info.current_h
    except:
        WIDTH = 1280
        HEIGHT = 720
    
    WIDTH = max(WIDTH, 800)
    HEIGHT = max(HEIGHT, 600)
    
    POS_HORIZONTAL_IZQUIERDA = int(WIDTH * 0.05)
    POS_HORIZONTAL_DERECHA = int(WIDTH * 0.95)
    RECTANGULO_HEIGHT = max(60, int(HEIGHT * 0.15))
    
    pong_sound = load_sound("pong.mpeg")
    win_sound = load_sound("win.mp3")
    countdown_sound = load_sound("countdown.mp3")
    
    left_paddle_y = HEIGHT // 2 - RECTANGULO_HEIGHT // 2
    right_paddle_y = HEIGHT // 2 - RECTANGULO_HEIGHT // 2
    
    reset_ball(direction=random.choice([-1, 1]))
    
    left_score = 0
    right_score = 0
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,  # Aumentada para mayor precisión
        min_tracking_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils

def get_hand_rect(hand_landmarks):
    x_coords = [lm.x * WIDTH for lm in hand_landmarks.landmark]
    y_coords = [lm.y * HEIGHT for lm in hand_landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    return (min_x, min_y), (max_x, max_y)

def draw_rectangle_hands(frame, hand_data):
    if not RECTANGLE_MODE:
        return
    
    try:
        for hand_label, rect in hand_data:
            color = (0, 255, 0)
            thickness = max(2, int(min(WIDTH, HEIGHT) * 0.003))
            
            # Dibujar rectángulo alrededor de la mano
            cv2.rectangle(frame, rect[0], rect[1], color, thickness)
            
            # Dibujar etiqueta de la mano
            font_scale = max(0.5, min(WIDTH, HEIGHT) / 1000)
            thickness_text = max(1, int(font_scale * 2))
            
            label_text = f"{hand_label} Hand"
            cv2.putText(frame, label_text,
                       (rect[0][0], rect[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness_text)
    except Exception as e:
        print(f"Error drawing hand rectangles: {e}")

def process_hands(results):
    global left_paddle_y, right_paddle_y, paused
    global COUNT_DOWN_LEFT_HAND, COUNT_DOWN_RIGHT_HAND
    
    hand_data = []
    left_hand_detected = False
    right_hand_detected = False

    if results.multi_hand_landmarks:
        hands_info = []
        for landmarks in results.multi_hand_landmarks:
            rect = get_hand_rect(landmarks)
            center_x = (rect[0][0] + rect[1][0]) // 2
            hands_info.append((center_x, rect))
        
        # Ordenar las manos por posición horizontal (de izquierda a derecha)
        hands_info.sort(key=lambda x: x[0])
        
        # Asignar etiquetas basadas en posición
        for i, (center_x, rect) in enumerate(hands_info):
            hand_label = 'Left' if i == 0 else 'Right'  # La mano más a la izquierda es Left, la otra Right
            hand_data.append((hand_label, rect))
            
            center_y = (rect[0][1] + rect[1][1]) // 2
            if hand_label == 'Left':
                left_hand_detected = True
                left_paddle_y = max(0, min(HEIGHT - RECTANGULO_HEIGHT, center_y - RECTANGULO_HEIGHT // 2))
            elif hand_label == 'Right':
                right_hand_detected = True
                right_paddle_y = max(0, min(HEIGHT - RECTANGULO_HEIGHT, center_y - RECTANGULO_HEIGHT // 2))
    
    # Actualizar contadores de detección con umbral más sensible
    if not left_hand_detected:
        COUNT_DOWN_LEFT_HAND = max(0, COUNT_DOWN_LEFT_HAND - 1/20)  # Decremento más rápido (1 segundo para llegar a 0)
    else:
        COUNT_DOWN_LEFT_HAND = min(2, COUNT_DOWN_LEFT_HAND + 1/10)  # Recuperación más lenta
    
    if not right_hand_detected:
        COUNT_DOWN_RIGHT_HAND = max(0, COUNT_DOWN_RIGHT_HAND - 1/20)
    else:
        COUNT_DOWN_RIGHT_HAND = min(2, COUNT_DOWN_RIGHT_HAND + 1/10)
    
    # Determinar estado de pausa
    paused = COUNT_DOWN_LEFT_HAND <= 0 or COUNT_DOWN_RIGHT_HAND <= 0
    
    # Debug: Mostrar valores de los contadores
    print(f"Left Hand Timer: {COUNT_DOWN_LEFT_HAND:.1f} | Right Hand Timer: {COUNT_DOWN_RIGHT_HAND:.1f} | Paused: {paused}")
    
    return hand_data, left_hand_detected, right_hand_detected

def draw_ball(frame):
    try:
        x, y = int(ballPosition[0]), int(ballPosition[1])
        ball_size = max(5, int(min(WIDTH, HEIGHT) * 0.015))
        cv2.circle(frame, (x, y), ball_size, (255, 255, 255), -1)
    except Exception as e:
        print(f"Error drawing ball: {e}")

def draw_middle_line(frame):
    try:
        line_thickness = max(1, int(min(WIDTH, HEIGHT) * 0.003))
        dash_length = max(5, int(HEIGHT * 0.03))
        gap_length = max(3, int(HEIGHT * 0.02))
        
        step = dash_length + gap_length
        if step <= 0:
            step = 10
            
        for y in range(0, HEIGHT, step):
            cv2.line(frame, (WIDTH // 2, y), (WIDTH // 2, y + dash_length), 
                    (255, 255, 255), line_thickness)
    except Exception as e:
        print(f"Error drawing middle line: {e}")

def draw_paddles(frame):
    try:
        paddle_width = max(10, int(WIDTH * 0.015))
        
        cv2.rectangle(frame, 
                     (POS_HORIZONTAL_IZQUIERDA - paddle_width // 2, left_paddle_y),
                     (POS_HORIZONTAL_IZQUIERDA + paddle_width // 2, left_paddle_y + RECTANGULO_HEIGHT),
                     (255, 255, 255), -1)
        
        cv2.rectangle(frame, 
                     (POS_HORIZONTAL_DERECHA - paddle_width // 2, right_paddle_y),
                     (POS_HORIZONTAL_DERECHA + paddle_width // 2, right_paddle_y + RECTANGULO_HEIGHT),
                     (255, 255, 255), -1)
    except Exception as e:
        print(f"Error drawing palettes: {e}")

def draw_score(frame):
    try:
        font_scale = max(0.5, min(WIDTH, HEIGHT) / 1000)
        thickness = max(1, int(font_scale * 2))
        
        # Dibujar marcador izquierdo
        left_text = f"Left: {left_score}"
        left_size = cv2.getTextSize(left_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        left_pos = (int(WIDTH * 0.1), int(HEIGHT * 0.1))
        cv2.putText(frame, left_text, left_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Mensaje de mano izquierda no detectada (siempre visible si hay problema)
        if COUNT_DOWN_LEFT_HAND < 2:  # Eliminamos la condición "and not paused"
            warning_text = "Left hand not detected!"
            warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, thickness)[0]
            cv2.putText(frame, warning_text,
                       (left_pos[0], left_pos[1] + left_size[1] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (0, 0, 255), thickness)
        
        # Dibujar marcador derecho
        right_text = f"Right: {right_score}"
        right_size = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        right_pos = (int(WIDTH * 0.7), int(HEIGHT * 0.1))
        cv2.putText(frame, right_text, right_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Mensaje de mano derecha no detectada (siempre visible si hay problema)
        if COUNT_DOWN_RIGHT_HAND < 2:  # Eliminamos la condición "and not paused"
            warning_text = "Right hand not detected!"
            warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, thickness)[0]
            cv2.putText(frame, warning_text,
                       (right_pos[0], right_pos[1] + right_size[1] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (0, 0, 255), thickness)
            
    except Exception as e:
        print(f"Error drawing marker: {e}")

def draw_shadow(frame):
    try:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    except Exception as e:
        print(f"Error drawing shadow: {e}")

def draw_pause_message(frame):
    try:
        # Solo dibujar el overlay oscuro si está en pausa
        if paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
            alpha = 0.3  # Reducimos la opacidad para que se vean mejor los otros mensajes
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Mensaje principal de pausa
            font_scale = max(0.8, min(WIDTH, HEIGHT) / 800)
            thickness = max(2, int(font_scale * 2))
            
            pause_text = "GAME PAUSED"
            text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.putText(frame, pause_text,
                       (WIDTH // 2 - text_size[0] // 2, HEIGHT // 2 - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
            
            continue_text = "Show both hands to continue"
            continue_size = cv2.getTextSize(continue_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale-0.2, thickness-1)[0]
            cv2.putText(frame, continue_text,
                       (WIDTH // 2 - continue_size[0] // 2, HEIGHT // 2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale-0.2, (255, 255, 255), thickness-1)
            
    except Exception as e:
        print(f"Error drawing pause message: {e}")

def draw_debug_info(frame):
    try:
        if not DEBUG_MODE:
            return
            
        font_scale = max(0.4, min(WIDTH, HEIGHT) / 1200)
        thickness = max(1, int(font_scale * 1.5))
        line_height = int(HEIGHT * 0.03)
        start_y = HEIGHT - 20
        
        debug_info = [
            f"Last Touched: {last_touched}",
            f"Game State: {'PAUSED' if paused else 'RUNNING'}",
            f"Right Hand Speed: {abs(ballSpeedX):.1f}",
            f"Left Hand Speed: {abs(ballSpeedX):.1f}",
            f"Ball Speed: ({abs(ballSpeedX):.1f}, {abs(ballSpeedY):.1f})",
            f"Ball Position: ({ballPosition[0]:.1f}, {ballPosition[1]:.1f})",
            f"Rectangle Mode: {'ON' if RECTANGLE_MODE else 'OFF'}"
        ]
        
        # Calcular el ancho máximo del texto
        text_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] for line in debug_info])
        
        # Fondo semitransparente para el texto de debug (esquina inferior izquierda)
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (10, start_y - len(debug_info) * line_height - 10),
                     (20 + text_width, start_y + 10),
                     (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Dibujar cada línea de información de debug
        for i, line in enumerate(debug_info):
            y_pos = start_y - (len(debug_info) - i - 1) * line_height
            cv2.putText(frame, line,
                       (15, y_pos),  # Posición X fija en 15 (izquierda)
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            
    except Exception as e:
        print(f"Error drawing debug info: {e}")

def speed_up():
    global ballSpeedX, ballSpeedY
    
    SPEED_INCREASE_FACTOR = 1.3
    
    ballSpeedX = abs(ballSpeedX) * SPEED_INCREASE_FACTOR * (1 if ballSpeedX > 0 else -1)
    ballSpeedY = abs(ballSpeedY) * SPEED_INCREASE_FACTOR * (1 if ballSpeedY > 0 else -1)
    
    if abs(ballSpeedX) > MAX_SPEED:
        ballSpeedX = MAX_SPEED * (1 if ballSpeedX > 0 else -1)
    
    if abs(ballSpeedY) > MAX_SPEED:
        ballSpeedY = MAX_SPEED * (1 if ballSpeedY > 0 else -1)

def update_ball_position(hand_data):
    global ballPosition, ballSpeedX, ballSpeedY, left_score, right_score, last_touched
    
    if paused:  # No actualizar posición si está en pausa
        return

    try:
        next_x = ballPosition[0] + ballSpeedX
        next_y = ballPosition[1] + ballSpeedY

        if next_y <= 0 or next_y >= HEIGHT:
            ballSpeedY = -ballSpeedY

        paddle_width = max(10, int(WIDTH * 0.015))
        
        if (POS_HORIZONTAL_IZQUIERDA - paddle_width//2 <= next_x <= POS_HORIZONTAL_IZQUIERDA + paddle_width//2 and 
            left_paddle_y <= next_y <= left_paddle_y + RECTANGULO_HEIGHT):
            if last_touched != 'Left':
                ballSpeedX = -ballSpeedX
                last_touched = 'Left'
                speed_up()
                if pong_sound:
                    channel = pygame.mixer.Channel(0)
                    channel.set_volume(1.0, 0.0)
                    channel.play(pong_sound)

        elif (POS_HORIZONTAL_DERECHA - paddle_width//2 <= next_x <= POS_HORIZONTAL_DERECHA + paddle_width//2 and 
              right_paddle_y <= next_y <= right_paddle_y + RECTANGULO_HEIGHT):
            if last_touched != 'Right':
                ballSpeedX = -ballSpeedX
                last_touched = 'Right'
                speed_up()
                if pong_sound:
                    channel = pygame.mixer.Channel(1)
                    channel.set_volume(0.0, 1.0)
                    channel.play(pong_sound)

        ballPosition[0] = next_x
        ballPosition[1] = next_y

        if ballPosition[0] <= 0:
            right_score += 1
            reset_ball(direction=-1)
            if win_sound:
                channel = pygame.mixer.Channel(2)
                channel.set_volume(0.0, 1.0)
                channel.play(win_sound)
        
        elif ballPosition[0] >= WIDTH:
            left_score += 1
            reset_ball(direction=1)
            if win_sound:
                channel = pygame.mixer.Channel(3)
                channel.set_volume(1.0, 0.0)
                channel.play(win_sound)
    except Exception as e:
        print(f"Error updating ball position: {e}")
        reset_ball(direction=1 if random.random() > 0.5 else -1)

def main():
    initialize_game()
    cap = None  # Inicializar la variable fuera del try
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("The camera could not be opened")
            
        cv2.namedWindow("Pong AR - Turn-Based", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Pong AR - Turn-Based", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error capturing frame")
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            hand_data, left_detected, right_detected = process_hands(results)
            
            draw_shadow(frame)
            draw_ball(frame)
            draw_middle_line(frame)
            draw_score(frame)
            draw_paddles(frame)
            draw_rectangle_hands(frame, hand_data)
            draw_debug_info(frame)
            
            if paused:
                draw_pause_message(frame)
            else:
                update_ball_position(hand_data)
            
            cv2.imshow("Pong AR - Turn-Based", frame)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                scape_to_menu()
                break
            elif key == 27:  # 27 es el código para la tecla ESC
                close()

    except KeyboardInterrupt:
        print("\nInterrupción por teclado detectada")
    except Exception as e:
        print(f"Error en el juego: {e}")
    finally:
        if cap is not None:  # Verificar si la cámara fue inicializada
            cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()
