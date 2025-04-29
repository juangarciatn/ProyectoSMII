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
from screeninfo import get_monitors

# Constantes de rutas
PONG = "assets/pong.mp3"
WIN = "assets/win.mp3"

try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    print("pygame.mixer inicializado correctamente")
except Exception as e:
    print(f"Error inicializando pygame.mixer: {e}")

pygame.mixer.music.set_volume(1.0)

def load_icon(path, size=(40, 40)):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: no se pudo cargar {path}")
        return None
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

icons = {
    "shield": load_icon("assets/shield.png"),
    "random": load_icon("assets/random.png"),
    "extra ball": load_icon("assets/speed.png"),
    "chaos": load_icon("assets/chaos.png")
}

def draw_paddles(frame):
    screen_ratio = SCREEN_WIDTH / SCREEN_HEIGHT
    original_ratio = WIDTH / HEIGHT
    
    if screen_ratio > original_ratio:
        scaled_height = int(RECTANGULO_HEIGHT * SCREEN_HEIGHT / HEIGHT)
        scaled_width = int(RECTANGULO_WIDTH * SCREEN_HEIGHT / HEIGHT)
    else:
        scaled_width = int(RECTANGULO_WIDTH * SCREEN_WIDTH / WIDTH)
        scaled_height = int(RECTANGULO_HEIGHT * SCREEN_WIDTH / WIDTH)
    
    min_paddle_size = int(10 * min(SCREEN_WIDTH/WIDTH, SCREEN_HEIGHT/HEIGHT))
    scaled_width = max(scaled_width, min_paddle_size)
    scaled_height = max(scaled_height, min_paddle_size)
    
    # Paleta izquierda (posición dinámica)
    left_y = left_paddle_y * (SCREEN_HEIGHT / HEIGHT)
    left_y = max(0, min(left_y, SCREEN_HEIGHT - scaled_height))
    cv2.rectangle(frame, (0, int(left_y)), 
                 (scaled_width, int(left_y + scaled_height)), 
                 (255, 255, 255), -1)
    
    # Paleta derecha (posición dinámica)
    right_y = right_paddle_y * (SCREEN_HEIGHT / HEIGHT)
    right_y = max(0, min(right_y, SCREEN_HEIGHT - scaled_height))
    cv2.rectangle(frame, (SCREEN_WIDTH - scaled_width, int(right_y)),
                 (SCREEN_WIDTH, int(right_y + scaled_height)), 
                 (255, 255, 255), -1)

def load_sound(filename):
    return pygame.mixer.Sound(filename) if os.path.exists(filename) else None

pong_sound = load_sound(PONG)
win_sound = load_sound(WIN)

WIDTH = 640
HEIGHT = 480
WINDOW_NAME = "PONG 2"
first_time = True

RECTANGULO_WIDTH = 10
RECTANGULO_HEIGHT = 90

left_paddle_y = HEIGHT // 2 - RECTANGULO_HEIGHT // 2
right_paddle_y = HEIGHT // 2 - RECTANGULO_HEIGHT // 2

BALL_SIZE = 10
INITIAL_BALL_EXTRA_BALL = 3
MAX_BALL_EXTRA_BALL = 15
HAND_EXTRA_BALL_MULTIPLIER = 5

balls = [{
    "pos": [WIDTH // 2, HEIGHT // 2],
    "vx": random.choice([-1, 1]) * INITIAL_BALL_EXTRA_BALL,
    "vy": random.choice([-1, 1]) * INITIAL_BALL_EXTRA_BALL,
    "last_touched": None
}]

MAX_BALLS = 5
left_score = 0
right_score = 0
last_touched = None

shields = {
    "left": False,
    "right": False
}

powerups = []
message_feed = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

last_detected_time = [0, 0]
warning_shown = [False, False]
game_active = False
game_paused = False

HISTORY_LENGTH = 5
SMOOTHING_ALPHA = 0.3

WARNING_TIME = 1.0
PAUSE_TIME = 3.0

POWERUP_RADIUS = 20
POWERUP_TYPES = ["shield", "random", "extra ball", "chaos"]
POWERUP_INTERVAL = 8
last_powerup_time = time.time()

MESSAGE_DURATION = 3.0

active_effects = {
    "left": None,
    "right": None
}

class HandTracker:
    def __init__(self):
        self.position_history = deque(maxlen=HISTORY_LENGTH)
        self.smoothed_speed = [0, 0]
        self.last_valid_position = None
        self.last_valid_time = time.time()
        self.hand_size = 50

    def update(self, new_position):
        current_time = time.time()
        if new_position:
            self.last_valid_position = new_position
            self.last_valid_time = current_time
            self.position_history.append((new_position, current_time))

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
                    self.smoothed_speed[0] = SMOOTHING_ALPHA * avg_speed_x + (1-SMOOTHING_ALPHA) * self.smoothed_speed[0]
                    self.smoothed_speed[1] = SMOOTHING_ALPHA * avg_speed_y + (1-SMOOTHING_ALPHA) * self.smoothed_speed[1]

        elif self.last_valid_position and (current_time - self.last_valid_time < 0.2):
            dt = current_time - self.last_valid_time
            estimated_x = self.last_valid_position[0] + self.smoothed_speed[0] * dt
            estimated_y = self.last_valid_position[1] + self.smoothed_speed[1] * dt
            return (estimated_x, estimated_y), self.smoothed_speed

        return new_position, self.smoothed_speed if new_position else (None, [0, 0])

hand_trackers = [HandTracker(), HandTracker()]

class AudioSystem:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {
            "left": self._load_sound(PONG),
            "right": self._load_sound(PONG),
            "wall": self._load_sound(PONG),
            "score": self._load_sound(WIN),
            "start": self._load_sound(WIN)
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
                self.channels["left"].set_volume(1.0, 0.0)
            elif sound_name == "right":
                self.channels["right"].play(self.sounds[sound_name])
                self.channels["right"].set_volume(0.0, 1.0)
            else:
                self.sounds[sound_name].play()

audio_system = AudioSystem()

def get_hand_rect(hand_landmarks):
    x_coords = [lm.x * WIDTH for lm in hand_landmarks.landmark]
    y_coords = [lm.y * HEIGHT for lm in hand_landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    return (min_x, min_y), (max_x, max_y)

def add_powerup_message(player_label, powerup_type):
    text = f"{player_label.capitalize()} got {powerup_type.upper()}!" if player_label != "SYSTEM" else powerup_type
    message_feed.append({"text": text, "start_time": time.time()})

def draw_powerup_feed(frame):
    current_time = time.time()
    message_feed[:] = [m for m in message_feed if current_time - m["start_time"] < MESSAGE_DURATION]

    font_scale = SCREEN_WIDTH / 1600
    thickness = int(2 * SCREEN_WIDTH / 800)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 255)

    base_y = int(50 * SCREEN_HEIGHT / HEIGHT)
    line_spacing = int(40 * SCREEN_HEIGHT / HEIGHT)

    for idx, msg in enumerate(message_feed):
        text_size = cv2.getTextSize(msg["text"], font, font_scale, thickness)[0]
        text_x = (SCREEN_WIDTH - text_size[0]) // 2
        text_y = base_y + idx * line_spacing
        cv2.putText(frame, msg["text"], (text_x, text_y), font, font_scale, color, thickness)

def draw_shadow(frame):
    try:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    except Exception as e:
        print(f"Error drawing shadow: {e}")

def process_hands(results):
    global last_detected_time, warning_shown, game_active, game_paused, left_paddle_y, right_paddle_y

    hand_data = []
    current_time = time.time()
    hands_detected = [False, False]
    middle_line = WIDTH // 2

    if results.multi_hand_landmarks:
        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            wrist_x = landmarks.landmark[mp_hands.HandLandmark.WRIST].x * WIDTH
            wrist_y = landmarks.landmark[mp_hands.HandLandmark.WRIST].y * HEIGHT

            if wrist_x < middle_line:
                label = "left"
                tracker_index = 0
            else:
                label = "right"
                tracker_index = 1

            rect = get_hand_rect(landmarks)
            center = ((rect[0][0] + rect[1][0]) / 2, (rect[0][1] + rect[1][1]) / 2)

            estimated_pos, speed = hand_trackers[tracker_index].update(center)

            if estimated_pos:
                if center is None:
                    estimated_pos = hand_trackers[tracker_index].last_valid_position

                hand_size = hand_trackers[tracker_index].hand_size
                min_x = int(estimated_pos[0] - hand_size)
                max_x = int(estimated_pos[0] + hand_size)
                min_y = int(estimated_pos[1] - hand_size)
                max_y = int(estimated_pos[1] + hand_size)

                hand_data.append((label, ((min_x, min_y), (max_x, max_y)), hand_trackers[tracker_index]))
                hands_detected[tracker_index] = True
                last_detected_time[tracker_index] = current_time

                # Actualizar posición de la paleta correspondiente
                hand_center_y = (min_y + max_y) // 2
                new_paddle_y = hand_center_y - RECTANGULO_HEIGHT // 2
                new_paddle_y = max(0, min(new_paddle_y, HEIGHT - RECTANGULO_HEIGHT))

                if label == "left":
                    left_paddle_y = new_paddle_y
                else:
                    right_paddle_y = new_paddle_y

                if warning_shown[tracker_index]:
                    warning_shown[tracker_index] = False

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
        norm_speed = min(np.sqrt(ball["vx"]**2 + ball["vy"]**2) / MAX_BALL_EXTRA_BALL, 1.0)
        color = (int(255 * (1 - norm_speed)), int(255 * (1 - norm_speed)), int(255 * norm_speed))
        cv2.circle(frame, center, scaled_size, color, -1)

def draw_shields(frame):
    thickness = int(10 * SCREEN_WIDTH / WIDTH)
    color = (0, 255, 255)

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

def reset_ball(ball=None, vx=None, vy=None):
    if ball is None:
        ball = {
            "pos": [WIDTH // 2, HEIGHT // 2],
            "vx": vx if vx is not None else random.choice([-1, 1]) * INITIAL_BALL_EXTRA_BALL,
            "vy": vy if vy is not None else random.choice([-1, 1]) * INITIAL_BALL_EXTRA_BALL,
            "last_touched": None
        }
        balls.append(ball)
    else:
        ball["pos"] = [WIDTH // 2, HEIGHT // 2]
        ball["vx"] = vx if vx is not None else random.choice([-1, 1]) * INITIAL_BALL_EXTRA_BALL
        ball["vy"] = vy if vy is not None else random.choice([-1, 1]) * INITIAL_BALL_EXTRA_BALL
        ball["last_touched"] = None

def update_all_balls(hand_data):
    global left_score, right_score, last_touched

    if game_paused or not game_active:
        return 0

    speeds = []

    for ball in balls[:]:
        pos = ball["pos"]
        vx = ball["vx"]
        vy = ball["vy"]

        next_x = pos[0] + vx
        next_y = pos[1] + vy

        # Rebote con los bordes superior e inferior
        if next_y <= 0 or next_y >= HEIGHT:
            vy = -vy
            if audio_system:
                audio_system.play("wall")

        collision = False
        
        # Colisión con las paletas (rectángulos blancos)
        # Paleta izquierda
        if (pos[0] - BALL_SIZE <= RECTANGULO_WIDTH and 
            left_paddle_y <= pos[1] <= left_paddle_y + RECTANGULO_HEIGHT):
            # Asegurar que la bola no se quede atrapada dentro de la paleta
            pos[0] = RECTANGULO_WIDTH + BALL_SIZE
            vx = abs(vx) * 1.1  # Invertir dirección X y aumentar velocidad
            vy = vy * 1.05  # Aumentar ligeramente la velocidad Y
            ball["last_touched"] = "left"
            if audio_system:
                audio_system.play("left")
            collision = True
        
        # Paleta derecha
        elif (pos[0] + BALL_SIZE >= WIDTH - RECTANGULO_WIDTH and 
              right_paddle_y <= pos[1] <= right_paddle_y + RECTANGULO_HEIGHT):
            # Asegurar que la bola no se quede atrapada dentro de la paleta
            pos[0] = WIDTH - RECTANGULO_WIDTH - BALL_SIZE
            vx = -abs(vx) * 1.1  # Invertir dirección X y aumentar velocidad
            vy = vy * 1.05  # Aumentar ligeramente la velocidad Y
            ball["last_touched"] = "right"
            if audio_system:
                audio_system.play("right")
            collision = True

        # Lógica de powerups (mantenemos esta parte)
        for label, ((min_x, min_y), (max_x, max_y)), _ in hand_data:
            for p in powerups[:]:  # Usamos una copia para poder modificarla durante la iteración
                if min_x < p["x"] < max_x and min_y < p["y"] < max_y:
                    powerup_type = p["type"]
                    powerups.remove(p)
                    add_powerup_message(label, powerup_type)

                    if powerup_type == "shield":
                        shields[label] = True
                    elif powerup_type == "random":
                        RANDOM_OFFSET_X = 40
                        for b in balls:
                            y = b["pos"][1]
                            if label == "left":
                                b["pos"][0] = WIDTH // 2 + RANDOM_OFFSET_X
                                b["vx"] = abs(b["vx"]) if b["vx"] is not None else INITIAL_BALL_EXTRA_BALL
                            else:
                                b["pos"][0] = WIDTH // 2 - RANDOM_OFFSET_X
                                b["vx"] = -abs(b["vx"]) if b["vx"] is not None else -INITIAL_BALL_EXTRA_BALL
                            b["vy"] = random.choice([-1, 1]) * random.uniform(2, 4)
                            b["last_touched"] = None

                    elif powerup_type == "extra ball":
                        if len(balls) < MAX_BALLS:
                            reset_ball()
                        elif len(balls) == MAX_BALLS:
                            add_powerup_message("SYSTEM", "MAXIMUM RANDOM")

        # Actualizar posición de la bola
        pos[0] = int(round(pos[0] + vx))
        pos[1] = int(round(pos[1] + vy))

        # Lógica de puntuación y escudos
        if pos[0] <= 0:
            if shields["left"]:
                shields["left"] = False
                pos[0] = 10
                ball["vx"] = abs(ball["vx"])
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
                pos[0] = WIDTH - 10
                ball["vx"] = -abs(ball["vx"])
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
    global left_pos, left_size, right_pos, right_size  # Add these global variables
    
    try:
        font_scale = max(0.5, min(SCREEN_WIDTH, SCREEN_HEIGHT) / 1000)
        thickness = max(1, int(font_scale * 2))
        
        left_text = f"Left: {left_score}"
        left_pos = scale_coords(50, 50)
        left_size = cv2.getTextSize(left_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.putText(frame, left_text, left_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        right_text = f"Right: {right_score}"
        right_pos = scale_coords(WIDTH - 200, 50)
        right_size = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.putText(frame, right_text, right_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
    except Exception as e:
        print(f"Error drawing marker: {e}")

def draw_warnings(frame):
    try:
        font_scale = max(0.5, min(SCREEN_WIDTH, SCREEN_HEIGHT) / 1000)
        thickness = max(1, int(font_scale * 2))
        
        if warning_shown[0]:
            warning_text = "Left hand not detected!"
            warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, thickness)[0]
            warning_pos = (left_pos[0], left_pos[1] + left_size[1] + int(15 * SCREEN_HEIGHT/HEIGHT))
            cv2.putText(frame, warning_text, warning_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (0, 0, 255), thickness)
        
        if warning_shown[1]:
            warning_text = "Right hand not detected!"
            warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, thickness)[0]
            warning_pos = (right_pos[0], right_pos[1] + right_size[1] + int(15 * SCREEN_HEIGHT/HEIGHT))
            cv2.putText(frame, warning_text, warning_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (0, 0, 255), thickness)
            
    except Exception as e:
        print(f"Error drawing warnings: {e}")

def draw_game_status(frame, hand_data):
    for label, ((min_x, min_y), (max_x, max_y)), _ in hand_data:
        scaled_min = scale_coords(min_x, min_y)
        scaled_max = scale_coords(max_x, max_y)
        cv2.rectangle(frame, scaled_min, scaled_max, (0, 255, 255), 1)

        text_pos = scale_coords(min_x + 5, min_y + 20)
        cv2.putText(frame, str(label), text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, SCREEN_WIDTH/1600,
                    (0, 255, 255), int(2 * SCREEN_WIDTH/800))

def draw_pause_message(frame):
    try:
        if game_paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0), -1)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            font_scale = max(0.8, min(SCREEN_WIDTH, SCREEN_HEIGHT) / 800)
            thickness = max(2, int(font_scale * 2))
            
            pause_text = "GAME PAUSED"
            text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.putText(frame, pause_text,
                       (SCREEN_WIDTH // 2 - text_size[0] // 2, SCREEN_HEIGHT // 2 - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
            
            continue_text = "Show both hands to continue"
            continue_size = cv2.getTextSize(continue_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale-0.2, thickness-1)[0]
            cv2.putText(frame, continue_text,
                       (SCREEN_WIDTH // 2 - continue_size[0] // 2, SCREEN_HEIGHT // 2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale-0.2, (255, 255, 255), thickness-1)
            
            instructions = [
                "'Q' --> Go to menu",
                "'ESC' --> Close game",
                "'R' --> Reset game"
            ]
            
            instruction_font_scale = font_scale - 0.3
            instruction_thickness = thickness - 1
            
            start_y = SCREEN_HEIGHT // 2 + 80
            
            for i, instruction in enumerate(instructions):
                text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 
                                          instruction_font_scale, instruction_thickness)[0]
                cv2.putText(frame, instruction,
                           (SCREEN_WIDTH // 2 - text_size[0] // 2, start_y + i * (text_size[1] + 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, instruction_font_scale, 
                           (200, 200, 200), instruction_thickness)
            
    except Exception as e:
        print(f"Error drawing pause message: {e}")

def draw_debug_info(frame):
    font_scale = SCREEN_WIDTH / 1600
    thickness = int(1 * SCREEN_WIDTH / 1600)
    line_height = int(25 * SCREEN_HEIGHT/HEIGHT)
    margin = int(5 * SCREEN_WIDTH/WIDTH)
    
    debug_texts = []

    for i, ball in enumerate(balls):
        pos = ball["pos"]
        vx = ball["vx"]
        vy = ball["vy"]
        debug_texts.append(f"Bola {i+1} Pos: ({pos[0]:.0f}, {pos[1]:.0f})  Vel: ({vx:.1f}, {vy:.1f})")

    debug_texts.append(f"Left Hand Speed: ({hand_trackers[0].smoothed_speed[0]:.1f}, {hand_trackers[0].smoothed_speed[1]:.1f})")
    debug_texts.append(f"Right Hand Speed: ({hand_trackers[1].smoothed_speed[0]:.1f}, {hand_trackers[1].smoothed_speed[1]:.1f})")
    debug_texts.append(f"Game State: {'Paused' if game_paused else 'Active' if game_active else 'Inactive'}")
    debug_texts.append(f"Bolas activas: {len(balls)}")

    max_text_width = 0
    for text in debug_texts:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        if text_size[0] > max_text_width:
            max_text_width = text_size[0]
    
    total_height = len(debug_texts) * line_height
    start_x = int(10 * SCREEN_WIDTH/WIDTH) - margin
    start_y = SCREEN_HEIGHT - total_height - int(30 * SCREEN_HEIGHT/HEIGHT) - margin
    
    overlay = frame.copy()
    cv2.rectangle(
        overlay, 
        (start_x, start_y), 
        (start_x + max_text_width + 2*margin, start_y + total_height + 2*margin), 
        (0, 0, 0), 
        -1
    )
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, text in enumerate(debug_texts):
        pos = (int(10 * SCREEN_WIDTH/WIDTH),
               SCREEN_HEIGHT - int(30 * SCREEN_HEIGHT/HEIGHT) - (i * line_height))
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 255), thickness)

def scale_coords(x, y):
    scale_x = SCREEN_WIDTH / WIDTH
    scale_y = SCREEN_HEIGHT / HEIGHT
    return (int(x * scale_x), int(y * scale_y))

try:
    monitor = get_monitors()[0]
    SCREEN_WIDTH = monitor.width
    SCREEN_HEIGHT = monitor.height
except:
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

def main(args):
    global INITIAL_BALL_EXTRA_BALL, INITIAL_BALL_SPEED, balls, left_paddle_y, right_paddle_y  # Añadir globales necesarias

    # Establecer la velocidad desde el argumento
    INITIAL_BALL_SPEED = args.velocidad
    INITIAL_BALL_EXTRA_BALL = args.velocidad

    # Reinicializar las bolas con la nueva velocidad
    balls.clear()
    balls.append({
        "pos": [WIDTH // 2, HEIGHT // 2],
        "vx": random.choice([-1, 1]) * INITIAL_BALL_SPEED,
        "vy": random.choice([-1, 1]) * INITIAL_BALL_SPEED,
        "last_touched": None
    })

    try:
        pygame.mixer.music.load("assets/arcade_acadia.mp3")
        volume = max(0.0, min(1.0, args.music_volume))
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

        current_time = time.time()
        if current_time - last_powerup_time > POWERUP_INTERVAL:
            x = WIDTH // 2
            y = random.randint(50, HEIGHT - 50)
            new_type = random.choice(POWERUP_TYPES)
            powerups.append({"x": x, "y": y, "type": new_type})
            last_powerup_time = current_time

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
        draw_shadow(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        hand_data = process_hands(results)

        draw_middle_line(frame)
        draw_shields(frame)
        draw_paddles(frame)
        draw_score(frame)

        for p in powerups:
            icon = icons.get(p["type"])
            if icon is not None:
                x, y = scale_coords(p["x"], p["y"])
                scale_factor = SCREEN_WIDTH / WIDTH
                w = int(icon.shape[1] * scale_factor * 1.5)
                h = int(icon.shape[0] * scale_factor * 1.5)
                resized_icon = cv2.resize(icon, (w, h), interpolation=cv2.INTER_CUBIC)
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = x1 + w, y1 + h

                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue

                roi = frame[y1:y2, x1:x2]
                alpha_icon = resized_icon[:, :, 3] / 255.0
                alpha_bg = 1.0 - alpha_icon

                for c in range(3):
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
            draw_pause_message(frame)

        cv2.imshow("Pong AR - Turn-Based", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            command = [sys.executable, "menu.py", "--music-volume", str(args.music_volume)]
            if 'debug' in args.extras:
                command.append("--debug")
            if 'rectangles' in args.extras:
                command.append("--rectangles")
            subprocess.Popen(command)
            break
        elif key == 27:
            break
        elif key == ord('r'):
            # global balls, left_score, right_score
            balls = [{
                "pos": [WIDTH // 2, HEIGHT // 2],
                "vx": random.choice([-1, 1]) * INITIAL_BALL_EXTRA_BALL,
                "vy": random.choice([-1, 1]) * INITIAL_BALL_EXTRA_BALL,
                "last_touched": None
            }]
            left_score = 0
            right_score = 0
            shields["left"] = False
            shields["right"] = False
            powerups.clear()
            message_feed.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pong 2 - AR Edition")
    parser.add_argument("--music-volume", type=float, default=0.5, help="Volumen de la música (0.0 a 1.0)")
    parser.add_argument("--debug", action="store_true", help="Activar modo debug")
    parser.add_argument("--rectangles", action="store_true", help="Mostrar rectángulos de detección")
    parser.add_argument("--velocidad", type=float, default=8.0, help="Velocidad inicial de la bola (1.0 a 15.0)")
    args = parser.parse_args()

    # Convertir los argumentos a un formato compatible
    args.extras = []
    if args.debug:
        args.extras.append("debug")
    if args.rectangles:
        args.extras.append("rectangles")

    main(args)
