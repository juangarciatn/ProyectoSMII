
import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
import random
import sys
import subprocess
import argparse
from collections import deque

try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    print("pygame.mixer inicializado correctamente")
except Exception as e:
    print(f"Error inicializando pygame.mixer: {e}")




# Añade esto después de inicializar el mixer
pygame.mixer.music.set_volume(1.0)  # Volumen global al máximo

def load_icon(path, size=(40, 40)):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Incluye canal alfa
    if img is None:
        print(f"Error: no se pudo cargar {path}")
        return None
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

icons = {
    "shield": load_icon("assets/shield.png"),
    "random": load_icon("assets/random.png"),
    "extra ball": load_icon("assets/speed.png")
}


def load_sound(filename):
    return pygame.mixer.Sound(filename) if os.path.exists(filename) else None

pong_sound = load_sound("pong.mp3")
win_sound = load_sound("win.mp3")

# Configuración de la ventana
WIDTH = 640
HEIGHT = 480

WINDOW_NAME = "PONG 2"

first_time = True

# Tamaño de la bola
BALL_SIZE = 10

# Configuración de velocidad de la bola
INITIAL_BALL_EXTRA_BALL = 3
MAX_BALL_EXTRA_BALL = 15
HAND_EXTRA_BALL_MULTIPLIER = 5

# Posición inicial de la bola
balls = [{
    "pos": [WIDTH // 2, HEIGHT // 2],
    "vx": INITIAL_BALL_EXTRA_BALL,
    "vy": INITIAL_BALL_EXTRA_BALL,
    "last_touched": None
}]

MAX_BALLS = 5



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


# Added Power-up system
powerups = []  # Lista de power-ups activos en pantalla
POWERUP_RADIUS = 20
POWERUP_TYPES = ["shield", "random", "extra ball"]
POWERUP_INTERVAL = 8  # Segundos entre power-ups
last_powerup_time = time.time()

# Mensajes activos (feed de power-ups)
message_feed = []  # Cada entrada será: {"text": str, "start_time": float}
MESSAGE_DURATION = 3.0  # Segundos


# Efectos activos para cada jugador
active_effects = {
    "left": None,
    "right": None
}

# Escudos activos
shields = {
    "left": False,
    "right": False
}


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
            "left": self._load_sound("pong.mpeg"),
            "right": self._load_sound("pong.mpeg"),
            "wall": self._load_sound("pong.mpeg"),
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

def add_powerup_message(player_label, powerup_type):
    if player_label == "SYSTEM":
        text = powerup_type  # Usa directamente el mensaje
    else:
        text = f"{player_label.capitalize()} got {powerup_type.upper()}!"

    message_feed.append({
        "text": text,
        "start_time": time.time()
    })


def draw_powerup_feed(frame):
    # Limpiar mensajes expirados
    current_time = time.time()
    message_feed[:] = [m for m in message_feed if current_time - m["start_time"] < MESSAGE_DURATION]

    font_scale = SCREEN_WIDTH / 1600
    thickness = int(2 * SCREEN_WIDTH / 800)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 255)  # Amarillo

    base_y = int(50 * SCREEN_HEIGHT / HEIGHT)
    line_spacing = int(40 * SCREEN_HEIGHT / HEIGHT)

    for idx, msg in enumerate(message_feed):
        text_size = cv2.getTextSize(msg["text"], font, font_scale, thickness)[0]
        text_x = (SCREEN_WIDTH - text_size[0]) // 2
        text_y = base_y + idx * line_spacing
        cv2.putText(frame, msg["text"], (text_x, text_y), font, font_scale, color, thickness)



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



def draw_ball(frame, speed):
    for ball in balls:
        center = scale_coords(int(round(ball["pos"][0])), int(round(ball["pos"][1])))
        scaled_size = int(BALL_SIZE * min(SCREEN_WIDTH/WIDTH, SCREEN_HEIGHT/HEIGHT))

        # Color según velocidad (simple)
        norm_speed = min(np.sqrt(ball["vx"]**2 + ball["vy"]**2) / MAX_BALL_EXTRA_BALL, 1.0)
        color = (int(255 * (1 - norm_speed)), int(255 * (1 - norm_speed)), int(255 * norm_speed))  # BGR

        cv2.circle(frame, center, scaled_size, color, -1)


def draw_shields(frame):
    thickness = int(10 * SCREEN_WIDTH / WIDTH)
    color = (0, 255, 255)  # Amarillo cian

    if shields["left"]:
        start = scale_coords(2.5, 0)
        end = scale_coords(2.5, HEIGHT)
        cv2.line(frame, start, end, color, thickness)

    if shields["right"]:
        start = scale_coords(WIDTH - 2.5, 0)
        end = scale_coords(WIDTH - 2.5, HEIGHT)
        cv2.line(frame, start, end, color, thickness)



def draw_middle_line(frame):
    line_thickness = int(2 * min(SCREEN_WIDTH/WIDTH, SCREEN_HEIGHT/HEIGHT))
    segment_length = int(20 * SCREEN_HEIGHT/HEIGHT)
    gap = int(10 * SCREEN_HEIGHT/HEIGHT)

    for y in range(0, SCREEN_HEIGHT, segment_length + gap):
        start_point = scale_coords(WIDTH // 2, y)
        end_point = scale_coords(WIDTH // 2, y + segment_length)
        if (y // gap) % 2 == 0:
            cv2.line(frame, start_point, end_point, (255, 255, 255), line_thickness)


# Asegúrate de que ballSpeedX y ballSpeedY no sean cero al comienzo
def reset_ball(ball=None, vx=None, vy=None):
    if ball is None:
        ball = {
            "pos": [WIDTH // 2, HEIGHT // 2],
            "vx": vx if vx is not None else random.choice([-1, 1]) * 3,
            "vy": vy if vy is not None else random.choice([-1, 1]) * 3,
            "last_touched": None
        }
        balls.append(ball)
    else:
        ball["pos"] = [WIDTH // 2, HEIGHT // 2]
        ball["vx"] = vx if vx is not None else random.choice([-1, 1]) * 3
        ball["vy"] = vy if vy is not None else random.choice([-1, 1]) * 3
        ball["last_touched"] = None




def update_all_balls(hand_data):
    global left_score, right_score, last_touched

    if game_paused or not game_active:
        return 0

    speeds = []

    for ball in balls[:]:  # Copia para poder eliminar
        pos = ball["pos"]
        vx = ball["vx"]
        vy = ball["vy"]

        next_x = pos[0] + vx
        next_y = pos[1] + vy

        # Rebote superior/inferior
        if next_y <= 0 or next_y >= HEIGHT:
            vy = -vy
            if audio_system:
                audio_system.play("wall")

        collision = False
        for label, ((min_x, min_y), (max_x, max_y)), _ in hand_data:
            for p in powerups:
                if min_x < p["x"] < max_x and min_y < p["y"] < max_y:
                    powerup_type = p["type"]
                    powerups.remove(p)
                    print(f"{label} obtuvo {powerup_type}")
                    add_powerup_message(label, powerup_type)

                    if powerup_type == "shield":
                        shields[label] = True
                    elif powerup_type == "random":
                        RANDOM_OFFSET_X = 40  # distancia desde el centro para reposicionar

                        for b in balls:
                            # Mantenemos la posición Y actual
                            y = b["pos"][1]

                            if label == "left":
                                # Aparece en campo derecho
                                b["pos"][0] = WIDTH // 2 + RANDOM_OFFSET_X
                                # Dirección hacia la portería enemiga (derecha): vx > 0
                                b["vx"] = abs(b["vx"]) if b["vx"] is not None else INITIAL_BALL_EXTRA_BALL
                            else:
                                # Aparece en campo izquierdo
                                b["pos"][0] = WIDTH // 2 - RANDOM_OFFSET_X
                                # Dirección hacia la portería enemiga (izquierda): vx < 0
                                b["vx"] = -abs(b["vx"]) if b["vx"] is not None else -INITIAL_BALL_EXTRA_BALL
                            print(f"RANDOM → pos: {b['pos']}, vx: {b['vx']}, vy: {b['vy']}")

                            # Restaurar la posición vertical
                            b["pos"][1] = y

                            # Velocidad vertical ligeramente aleatoria
                            b["vy"] = random.choice([-1, 1]) * random.uniform(2, 4)
                            b["last_touched"] = None  # permitir que cualquiera golpee después





                    elif powerup_type == "extra ball":
                        if len(balls) < MAX_BALLS:
                            reset_ball()
                            print("¡Nueva bola generada por EXTRA BALL!")
                        elif len(balls) == MAX_BALLS:
                            add_powerup_message("SYSTEM", "MAXIMUM RANDOM")


            if (min_x - BALL_SIZE <= next_x <= max_x + BALL_SIZE and
                min_y - BALL_SIZE <= next_y <= max_y + BALL_SIZE):

                overlap_left = abs(next_x - min_x)
                overlap_right = abs(next_x - max_x)
                overlap_top = abs(next_y - min_y)
                overlap_bottom = abs(next_y - max_y)
                min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

                if ball["last_touched"] != label:
                    if min_overlap == overlap_left or min_overlap == overlap_right:
                        vx = -vx * 1.1
                        vy = random.uniform(2, 5) * random.choice([-1, 1])
                        if audio_system:
                            audio_system.play(label)
                    elif min_overlap == overlap_top or min_overlap == overlap_bottom:
                        vy = -vy
                        vx += random.uniform(-1, 1)
                        if audio_system:
                            audio_system.play("wall")

                    speed_magnitude = np.sqrt(vx**2 + vy**2)
                    if speed_magnitude > MAX_BALL_EXTRA_BALL:
                        scale = MAX_BALL_EXTRA_BALL / speed_magnitude
                        vx *= scale
                        vy *= scale

                    ball["last_touched"] = label
                    collision = True
                    break

        # Actualizar posición
        pos[0] = int(round(pos[0] + vx))
        pos[1] = int(round(pos[1] + vy))

        # Puntos
        if pos[0] <= 0:
            if shields["left"]:
                shields["left"] = False
                print("¡Escudo izquierdo activado!")
                pos[0] = 10  # rebota ligeramente dentro del campo
                ball["vx"] = abs(ball["vx"])  # rebote hacia la derecha
                if audio_system:
                    audio_system.play("wall")
            else:
                right_score += 1
                balls.remove(ball)
                if len(balls) == 0:
                    reset_ball()
                if audio_system:
                    audio_system.play("score")

        elif pos[0] >= WIDTH:
            if shields["right"]:
                shields["right"] = False
                print("¡Escudo derecho activado!")
                pos[0] = WIDTH - 10
                ball["vx"] = -abs(ball["vx"])  # rebote hacia la izquierda
                if audio_system:
                    audio_system.play("wall")
            else:
                left_score += 1
                balls.remove(ball)
                if len(balls) == 0:
                    reset_ball()
                if audio_system:
                    audio_system.play("score")


        else:
            ball["vx"], ball["vy"] = vx, vy
            speeds.append(np.sqrt(vx**2 + vy**2))

    return max(speeds) if speeds else 0





def draw_score(frame):
    font_scale = SCREEN_WIDTH / 800
    thickness = int(2 * SCREEN_WIDTH / 800)

    left_text = f"Left: {left_score}"
    right_text = f"Right: {right_score}"

    # Posición izquierda escalada
    left_pos = scale_coords(50, 50)
    cv2.putText(frame, left_text, left_pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 255, 255), thickness)

    # Posición derecha escalada
    right_pos = scale_coords(WIDTH - 200, 50)
    cv2.putText(frame, right_text, right_pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 255, 255), thickness)

def draw_warnings(frame):
    font_scale = SCREEN_WIDTH / 1600
    thickness = int(1 * SCREEN_WIDTH / 800)

    for i in range(2):
        if warning_shown[i]:
            hand_name = "Left" if i == 0 else "Right"
            
            # Posición base del marcador correspondiente
            if i == 0:  # Left
                warning_pos = scale_coords(50, 80)  # Debajo de "Left: X"
            else:  # Right
                warning_pos = scale_coords(WIDTH - 200, 80)  # Debajo de "Right: X"
                
            cv2.putText(frame, f"{hand_name} hand not detected!",
                       warning_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (0, 255, 255), thickness)

def draw_game_status(frame, hand_data):
    for label, ((min_x, min_y), (max_x, max_y)), _ in hand_data:
        # Escalar las coordenadas del rectángulo
        scaled_min = scale_coords(min_x, min_y)
        scaled_max = scale_coords(max_x, max_y)
        cv2.rectangle(frame, scaled_min, scaled_max, (0, 255, 255), 1)

        # Escalar posición del texto
        text_pos = scale_coords(min_x + 5, min_y + 20)
        cv2.putText(frame, str(label), text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, SCREEN_WIDTH/1600,
                    (0, 255, 255), int(2 * SCREEN_WIDTH/800))


def draw_debug_info(frame):
    font_scale = SCREEN_WIDTH / 1600
    thickness = int(1 * SCREEN_WIDTH / 1600)
    line_height = int(25 * SCREEN_HEIGHT/HEIGHT)
    margin = int(5 * SCREEN_WIDTH/WIDTH)
    
    debug_texts = []

    # Mostrar info de cada bola
    for i, ball in enumerate(balls):
        pos = ball["pos"]
        vx = ball["vx"]
        vy = ball["vy"]
        debug_texts.append(f"Bola {i+1} Pos: ({pos[0]:.0f}, {pos[1]:.0f})  Vel: ({vx:.1f}, {vy:.1f})")

    # Info de las manos
    debug_texts.append(f"Left Hand Speed: ({hand_trackers[0].smoothed_speed[0]:.1f}, {hand_trackers[0].smoothed_speed[1]:.1f})")
    debug_texts.append(f"Right Hand Speed: ({hand_trackers[1].smoothed_speed[0]:.1f}, {hand_trackers[1].smoothed_speed[1]:.1f})")

    # Estado del juego
    debug_texts.append(f"Game State: {'Paused' if game_paused else 'Active' if game_active else 'Inactive'}")
    debug_texts.append(f"Bolas activas: {len(balls)}")

    # Calcular tamaño del rectángulo de fondo
    max_text_width = 0
    for text in debug_texts:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        if text_size[0] > max_text_width:
            max_text_width = text_size[0]
    
    total_height = len(debug_texts) * line_height
    start_x = int(10 * SCREEN_WIDTH/WIDTH) - margin
    start_y = SCREEN_HEIGHT - total_height - int(30 * SCREEN_HEIGHT/HEIGHT) - margin
    
    # Crear rectángulo semitransparente
    overlay = frame.copy()
    cv2.rectangle(
        overlay, 
        (start_x, start_y), 
        (start_x + max_text_width + 2*margin, start_y + total_height + 2*margin), 
        (0, 0, 0), 
        -1
    )
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Mostrar los textos en pantalla
    for i, text in enumerate(debug_texts):
        pos = (int(10 * SCREEN_WIDTH/WIDTH),
               SCREEN_HEIGHT - int(30 * SCREEN_HEIGHT/HEIGHT) - (i * line_height))
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 255), thickness)

# Función auxiliar para escalar coordenadas
def scale_coords(x, y):
    scale_x = SCREEN_WIDTH / WIDTH
    scale_y = SCREEN_HEIGHT / HEIGHT
    return (int(x * scale_x), int(y * scale_y))

from screeninfo import get_monitors
import cv2
import sys

# Obtener la resolución de la pantalla principal
try:
    monitor = get_monitors()[0]
    SCREEN_WIDTH = monitor.width
    SCREEN_HEIGHT = monitor.height
    scale_factor = SCREEN_WIDTH / WIDTH  # Escala proporcional al ancho
    w = int(icon.shape[1] * scale_factor * 1.5)  # El 1.5 es para agrandarlo aún más
    h = int(icon.shape[0] * scale_factor * 1.5)
    resized_icon = cv2.resize(icon, (w, h), interpolation=cv2.INTER_AREA)

except:
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

def main(args):
    try:
        pygame.mixer.music.load("assets/arcade_acadia.mp3")
        volume = max(0.0, min(1.0, args.music_volume))  # asegura que esté entre 0.0 y 1.0
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(-1)
        print(f"🎵 Música cargada con volumen {volume}")
    except Exception as e:
        print(f"Error al cargar música: {e}")

    last_powerup_time = time.time()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)


    cv2.namedWindow("Pong AR - Turn-Based", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pong AR - Turn-Based", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame")
            break
        # Added Aparecer nuevo power-up si ha pasado tiempo suficiente
        current_time = time.time()
        if current_time - last_powerup_time > POWERUP_INTERVAL:
            x = WIDTH // 2
            y = random.randint(50, HEIGHT - 50)
            new_type = random.choice(POWERUP_TYPES)
            powerups.append({"x": x, "y": y, "type": new_type})
            last_powerup_time = current_time


        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))  # <<--- ESTA ES LA LÍNEA NUEVA

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        hand_data = process_hands(results)

        draw_middle_line(frame)
        draw_score(frame)
        draw_shields(frame)
        #added
        for p in powerups:
            icon = icons.get(p["type"])
            if icon is not None:
                x, y = scale_coords(p["x"], p["y"])

                # 👇 Calcula tamaño dinámico del icono
                scale_factor = SCREEN_WIDTH / WIDTH
                w = int(icon.shape[1] * scale_factor * 1.5)
                h = int(icon.shape[0] * scale_factor * 1.5)
                resized_icon = cv2.resize(icon, (w, h), interpolation=cv2.INTER_CUBIC)

                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = x1 + w, y1 + h

                # Asegúrate de no salirte del frame
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue

                roi = frame[y1:y2, x1:x2]
                alpha_icon = resized_icon[:, :, 3] / 255.0
                alpha_bg = 1.0 - alpha_icon

                for c in range(3):  # BGR
                    roi[:, :, c] = (alpha_icon * resized_icon[:, :, c] +
                                    alpha_bg * roi[:, :, c])


        speed = update_all_balls(hand_data)
        draw_ball(frame, speed)
        draw_warnings(frame)
        draw_powerup_feed(frame)

        if 'rectangles' in args.extras:
            draw_game_status(frame, hand_data)

        if 'debug' in args.extras:
            draw_debug_info(frame)


        if not game_active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            text = "Waiting for 2 players..."
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                    SCREEN_WIDTH/800, 2)[0]
            text_pos = ((SCREEN_WIDTH - text_size[0])//2, SCREEN_HEIGHT//2)
            cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        SCREEN_WIDTH/800, (0, 0, 255), 2)
        elif game_paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # Mensaje principal
            main_text = "Game Paused - Show both hands to resume"
            text_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX,
                                        SCREEN_WIDTH/1600, 2)[0]
            text_pos = ((SCREEN_WIDTH - text_size[0])//2, SCREEN_HEIGHT//2 - 50)
            cv2.putText(frame, main_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        SCREEN_WIDTH/1600, (0, 255, 255), 2)
    
            # Mensajes de controles
            controls = [
                "'Q' --> Go to menu",
                "'ESC' --> Close game",
                "'R' --> Reset game"
            ]
    
            # Espaciado entre líneas
            line_spacing = 40

            for i, control_text in enumerate(controls):
                control_size = cv2.getTextSize(control_text, cv2.FONT_HERSHEY_SIMPLEX,
                                        SCREEN_WIDTH/2000, 1)[0]
                control_pos = ((SCREEN_WIDTH - control_size[0])//2, 
                              SCREEN_HEIGHT//2 + i * line_spacing)
                cv2.putText(frame, control_text, control_pos, cv2.FONT_HERSHEY_SIMPLEX,
                            SCREEN_WIDTH/2000, (255, 255, 255), 1)


        cv2.imshow("Pong AR - Turn-Based", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Leaving pong2.py...")
            print("Going to menu.py...")
            subprocess.Popen([sys.executable, "menu.py"])
            break
        elif key == 27:  # 27 es el código ASCII para ESC
            print("Closing pong2.py...")
            break
        elif key == ord('r'):
            speed = np.sqrt(ballSpeedX ** 2 + ballSpeedY ** 2)
            print(f"Reiniciando pelota con velocidad: {speed:.2f}")
            reset_ball(speed)
            last_touched = None


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pong 2 - AR Edition")
    parser.add_argument("--music-volume", type=float, default=0.5, help="Volumen de la música (0.0 a 1.0)")
    parser.add_argument("extras", nargs="*", help="Argumentos extra como 'debug' o 'rectangles'")
    args = parser.parse_args()
    main(args)


