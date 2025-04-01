import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
import random
from collections import deque

try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    print("pygame.mixer inicializado correctamente")
except Exception as e:
    print(f"Error inicializando pygame.mixer: {e}")


# Añade esto después de inicializar el mixer
pygame.mixer.music.set_volume(1.0)  # Volumen global al máximo

# Verifica que los archivos existan en estas rutas exactas
print("Verificando archivos:")
print(f"Existe pong.mp3: {os.path.exists('pong.mp3')}")
print(f"Existe win.mp3: {os.path.exists('win.mp3')}")

def load_sound(filename):
    return pygame.mixer.Sound(filename) if os.path.exists(filename) else None

pong_sound = load_sound("pong.mp3")
win_sound = load_sound("win.mp3")

# Configuración de la ventana
WIDTH = 640
HEIGHT = 480

# Tamaño de la bola
BALL_SIZE = 10

# Configuración de velocidad de la bola
INITIAL_BALL_SPEED = 5
MAX_BALL_SPEED = 15
HAND_SPEED_MULTIPLIER = 5

# Posición inicial de la bola
ballPosition = [int(WIDTH // 2), int(HEIGHT // 2)]
ballSpeedX = INITIAL_BALL_SPEED
ballSpeedY = INITIAL_BALL_SPEED

# Puntuación
left_score = 0
right_score = 0
last_touched = None  # 1: izquierda, 2: derecha

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables para el seguimiento
last_detected_time = [0, 0]  # Para mano izquierda (0) y derecha (1)
warning_shown = [False, False]  # Para controlar si se mostró advertencia
game_active = False
game_paused = False

# Configuración del sistema de seguimiento mejorado
HISTORY_LENGTH = 5  # Número de frames a considerar para el promedio móvil
SMOOTHING_ALPHA = 0.3  # Factor de suavizado exponencial

WARNING_TIME = 1.0
PAUSE_TIME = 3.0

class HandTracker:
    def __init__(self):
        self.position_history = deque(maxlen=HISTORY_LENGTH)
        self.smoothed_speed = [0, 0]
        self.last_valid_position = None
        self.last_valid_time = time.time()
        self.hand_size = 50  # Tamaño estimado de la mano en píxeles

    def update(self, new_position):
        current_time = time.time()

        # Si tenemos una nueva posición válida
        if new_position:
            self.last_valid_position = new_position
            self.last_valid_time = current_time
            self.position_history.append((new_position, current_time))

            # Calcular velocidad basada en el historial
            if len(self.position_history) >= 2:
                total_dx, total_dy, total_time = 0, 0, 0
                for i in range(1, len(self.position_history)):
                    prev_pos, prev_time = self.position_history[i-1]
                    curr_pos, curr_time = self.position_history[i]
                    total_dx += curr_pos[0] - prev_pos[0]
                    total_dy += curr_pos[1] - prev_pos[1]
                    total_time += curr_time - prev_time

                if total_time > 0:
                    avg_speed_x = total_dx / total_time
                    avg_speed_y = total_dy / total_time
                    # Aplicar suavizado exponencial
                    self.smoothed_speed[0] = SMOOTHING_ALPHA * avg_speed_x + (1-SMOOTHING_ALPHA) * self.smoothed_speed[0]
                    self.smoothed_speed[1] = SMOOTHING_ALPHA * avg_speed_y + (1-SMOOTHING_ALPHA) * self.smoothed_speed[1]

        # Si no hay detección pero tenemos datos anteriores, estimar posición
        elif self.last_valid_position and (current_time - self.last_valid_time < 0.2):  # 200ms de tolerancia
            dt = current_time - self.last_valid_time
            estimated_x = self.last_valid_position[0] + self.smoothed_speed[0] * dt
            estimated_y = self.last_valid_position[1] + self.smoothed_speed[1] * dt
            return (estimated_x, estimated_y), self.smoothed_speed

        return new_position, self.smoothed_speed if new_position else (None, [0, 0])

# Inicializar trackers para cada mano
hand_trackers = [HandTracker(), HandTracker()]

import pygame
import numpy as np

import pygame

class AudioSystem:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {
            "left": self._load_sound("pong.mp3"),
            "right": self._load_sound("pong.mp3"),
            "wall": self._load_sound("pong.mp3"),
            "score": self._load_sound("win.mp3"),
            "start": self._load_sound("win.mp3")
        }
        self.channels = {
            "left": pygame.mixer.Channel(0),
            "right": pygame.mixer.Channel(1)
        }

    def _load_sound(self, path):
        try:
            return pygame.mixer.Sound(path)
        except pygame.error as e:
            print(f"Error cargando {path}: {e}")
            return None

    def play(self, sound_name):
        if sound_name in self.sounds and self.sounds[sound_name]:
            if sound_name == "left":
                self.channels["left"].play(self.sounds[sound_name])
                self.channels["left"].set_volume(1.0, 0.0)  # Solo canal izquierdo
            elif sound_name == "right":
                self.channels["right"].play(self.sounds[sound_name])
                self.channels["right"].set_volume(0.0, 1.0)  # Solo canal derecho
            else:
                self.sounds[sound_name].play()  # Otros sonidos normales
        else:
            print(f"Advertencia: sonido '{sound_name}' no disponible.")



# Inicialización del sistema de audio
audio_system = AudioSystem()

# Prueba de sonido si los archivos están disponibles
if pong_sound:
    pong_sound.play()
if win_sound:
    win_sound.play()

# Inicialización compatible
audio_system = AudioSystem()

# Añade esto después de crear audio_system
test_sound = pygame.mixer.Sound(buffer=bytes([128, 255] * 1024))  # Sonido de prueba
test_sound.play()

def get_hand_rect(hand_landmarks):
    x_coords = [lm.x * WIDTH for lm in hand_landmarks.landmark]
    y_coords = [lm.y * HEIGHT for lm in hand_landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    return (min_x, min_y), (max_x, max_y)

def process_hands(results):
    global last_detected_time, warning_shown, game_active, game_paused

    hand_data = []
    current_time = time.time()
    hands_detected = [False, False]  # [left, right]
    middle_line = WIDTH // 2  # Línea central de la pantalla

    if results.multi_hand_landmarks:
        # Procesamos cada mano detectada
        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Obtenemos la posición de la muñeca como referencia
            wrist_x = landmarks.landmark[mp_hands.HandLandmark.WRIST].x * WIDTH
            wrist_y = landmarks.landmark[mp_hands.HandLandmark.WRIST].y * HEIGHT

            # Determinamos si es mano izquierda o derecha basado en posición
            if wrist_x < middle_line:
                label = "left"
                tracker_index = 0
            else:
                label = "right"
                tracker_index = 1

            # Obtenemos el rectángulo delimitador
            rect = get_hand_rect(landmarks)
            center = ((rect[0][0] + rect[1][0]) / 2, (rect[0][1] + rect[1][1]) / 2)

            # Actualizamos el tracker correspondiente
            estimated_pos, speed = hand_trackers[tracker_index].update(center)

            if estimated_pos:
                # Si la posición es estimada, usar la última válida
                if center is None:
                    estimated_pos = hand_trackers[tracker_index].last_valid_position

                # Reconstruir rectángulo basado en posición estimada
                hand_size = hand_trackers[tracker_index].hand_size
                min_x = int(estimated_pos[0] - hand_size)
                max_x = int(estimated_pos[0] + hand_size)
                min_y = int(estimated_pos[1] - hand_size)
                max_y = int(estimated_pos[1] + hand_size)

                hand_data.append((label, ((min_x, min_y), (max_x, max_y)), hand_trackers[tracker_index]))
                hands_detected[tracker_index] = True
                last_detected_time[tracker_index] = current_time

                if warning_shown[tracker_index]:
                    warning_shown[tracker_index] = False

    # Resto del código para manejar detecciones perdidas...
    for i in range(2):
        time_since_last_detection = current_time - last_detected_time[i]

        if not hands_detected[i] and last_detected_time[i] > 0:
            if time_since_last_detection > WARNING_TIME and not warning_shown[i]:
                warning_shown[i] = True

            if time_since_last_detection > PAUSE_TIME:
                game_paused = True

    if all(hands_detected) and (not game_active or game_paused):
        game_active = True
        game_paused = False
        for i in range(2):
            warning_shown[i] = False

    return hand_data

def draw_ball(frame):
    center = (int(round(ballPosition[0])), int(round(ballPosition[1])))
    cv2.circle(frame, center, BALL_SIZE, (255, 255, 255), -1)

def draw_middle_line(frame):
    for y in range(0, HEIGHT, 20):
        if (y // 10) % 2 == 0:
            cv2.line(frame, (WIDTH // 2, y), (WIDTH // 2, y + 10), (255, 255, 255), 2)

import random

def reset_ball():
    global ballPosition, ballSpeedX, ballSpeedY, last_touched

    # Posición inicial centrada
    ballPosition = [WIDTH // 2, HEIGHT // 2]

    # Velocidad inicial aleatoria
    ballSpeedX = random.choice([-4, 4])
    ballSpeedY = random.uniform(-2, 2)

    # Reinicia quién tocó último
    last_touched = None

    # Sonido de inicio
    if audio_system:
        audio_system.play("start")

def update_ball_position(hand_data):
    global ballPosition, ballSpeedX, ballSpeedY, left_score, right_score, last_touched

    if game_paused or not game_active:
        return

    next_x = float(ballPosition[0]) + ballSpeedX
    next_y = float(ballPosition[1]) + ballSpeedY

    # Rebotar en bordes superior/inferior
    if next_y <= 0 or next_y >= HEIGHT:
        ballSpeedY = -ballSpeedY
        if audio_system:
            audio_system.play("wall")

    # Verificar colisiones con las manos
    collision = False
    for label, ((min_x, min_y), (max_x, max_y)), hand_tracker in hand_data:
        if (min_x - BALL_SIZE <= next_x <= max_x + BALL_SIZE and
            min_y - BALL_SIZE <= next_y <= max_y + BALL_SIZE):

            overlap_left = abs(next_x - min_x)
            overlap_right = abs(next_x - max_x)
            overlap_top = abs(next_y - min_y)
            overlap_bottom = abs(next_y - max_y)
            min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

            if last_touched != label:
                if min_overlap == overlap_left or min_overlap == overlap_right:
                    ballSpeedX = -ballSpeedX * 1.1
                    random_direction = random.choice([-1, 1])
                    ballSpeedY = random.uniform(2, 5) * random_direction

                    if audio_system:
                        audio_system.play(label)

                elif min_overlap == overlap_top or min_overlap == overlap_bottom:
                    ballSpeedY = -ballSpeedY
                    ballSpeedX += random.uniform(-1, 1)

                    if audio_system:
                        audio_system.play("wall")

                if (ballSpeedX > 0 and next_x < min_x) or (ballSpeedX < 0 and next_x > max_x):
                    ballSpeedX = -ballSpeedX

                speed_magnitude = np.sqrt(ballSpeedX**2 + ballSpeedY**2)
                if speed_magnitude > MAX_BALL_SPEED:
                    scale_factor = MAX_BALL_SPEED / speed_magnitude
                    ballSpeedX *= scale_factor
                    ballSpeedY *= scale_factor

                last_touched = label
                collision = True
                break

    # Actualizar posición
    ballPosition[0] = int(round(next_x))
    ballPosition[1] = int(round(next_y))

    # Puntos cuando la pelota sale de la pantalla
    if ballPosition[0] <= 0:
        right_score += 1
        last_touched = None
        if audio_system:
            audio_system.play("score")
        reset_ball()
    elif ballPosition[0] >= WIDTH:
        left_score += 1
        last_touched = None
        if audio_system:
            audio_system.play("score")
        reset_ball()

def reset_ball():
    global ballPosition, ballSpeedX, ballSpeedY
    ballPosition = [int(WIDTH // 2), int(HEIGHT // 2)]
    ballSpeedX = INITIAL_BALL_SPEED if np.random.rand() > 0.5 else -INITIAL_BALL_SPEED
    ballSpeedY = INITIAL_BALL_SPEED

def draw_score(frame):
    cv2.putText(frame, f"Left: {left_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Right: {right_score}", (WIDTH - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_warnings(frame):
    for i in range(2):
        if warning_shown[i]:
            hand_name = "Left" if i == 0 else "Right"
            cv2.putText(frame, f"{hand_name} hand not detected!",
                        (20, 100 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def draw_game_status(frame):
    if not game_active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "Waiting for 2 players...",
                    (WIDTH//4, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif game_paused:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "Game Paused - Show both hands to resume",
                    (WIDTH//6, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def draw_debug_info(frame):
    # Mostrar información de debug en la esquina inferior izquierda
    debug_texts = [
        f"Ball Position: ({ballPosition[0]}, {ballPosition[1]})",
        f"Ball Speed: ({ballSpeedX:.1f}, {ballSpeedY:.1f})",
        f"Left Hand Speed: ({hand_trackers[0].smoothed_speed[0]:.1f}, {hand_trackers[0].smoothed_speed[1]:.1f})",
        f"Right Hand Speed: ({hand_trackers[1].smoothed_speed[0]:.1f}, {hand_trackers[1].smoothed_speed[1]:.1f})",
        f"Game State: {'Active' if game_active else 'Inactive'} {'(Paused)' if game_paused else ''}",
        f"Last Touched: {'Left' if last_touched == 1 else 'Right' if last_touched == 2 else 'None'}"
    ]

    for i, text in enumerate(debug_texts):
        cv2.putText(frame, text, (10, HEIGHT - 30 - (i * 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar manos en cada frame
        results = hands.process(rgb_frame)
        hand_data = process_hands(results)

        # Dibujar elementos del juego
        draw_ball(frame)
        draw_middle_line(frame)
        draw_score(frame)
        update_ball_position(hand_data)
        draw_warnings(frame)
        draw_game_status(frame)
        draw_debug_info(frame)

        # Dibujar rectángulos y etiquetas de manos
        if game_active or not game_paused:
            for label, ((min_x, min_y), (max_x, max_y)), _ in hand_data:
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 1)
                cv2.putText(frame, str(label), (min_x + 5, min_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Pong AR - Turn-Based", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
