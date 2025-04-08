import cv2
import mediapipe as mp
import numpy as np
import os
import time
import wave
from openal import alc, al
from ctypes import c_uint, c_void_p, byref

# Configuración inicial de OpenAL
device = alc.alcOpenDevice(None)
context = alc.alcCreateContext(device, None)
alc.alcMakeContextCurrent(context)

# Configuraciones del juego
WIDTH, HEIGHT = 640, 480
RECTANGULO_WIDTH, RECTANGULO_HEIGHT = 30, 90
POS_HORIZONTAL_IZQUIERDA = 30
POS_HORIZONTAL_DERECHA = 610
BALL_SIZE = 10
COUNT_DOWN = 3

# Variables del juego
ballPosition = [WIDTH // 2, HEIGHT // 2]
ballSpeedX, ballSpeedY = 5, 5
left_score, right_score = 0, 0
last_touched = None
left_paddle_y = HEIGHT // 2 - RECTANGULO_HEIGHT // 2
right_paddle_y = HEIGHT // 2 - RECTANGULO_HEIGHT // 2

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

def load_sound(filename):
    if not os.path.exists(filename):
        return None
    
    with wave.open(filename, 'rb') as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()  # Obtener profundidad de bits (1=8bits, 2=16bits)
        sample_rate = wav_file.getframerate()
        data = wav_file.readframes(wav_file.getnframes())
    
    buffer_id = c_uint(0)
    al.alGenBuffers(1, byref(buffer_id))
    
    # Determinar formato según canales y profundidad
    if sample_width == 1:
        format = al.AL_FORMAT_STEREO8 if channels == 2 else al.AL_FORMAT_MONO8
    else:  # 16 bits (sample_width == 2)
        format = al.AL_FORMAT_STEREO16 if channels == 2 else al.AL_FORMAT_MONO16
    
    al.alBufferData(buffer_id.value, format, data, len(data), sample_rate)
    
    return buffer_id

def play_sound(buffer_id, pan=0.0):
    if not buffer_id:
        return
    
    source_id = c_uint(0)
    al.alGenSources(1, byref(source_id))
    al.alSourcei(source_id, al.AL_BUFFER, buffer_id.value)
    al.alSource3f(source_id, al.AL_POSITION, pan, 0, 0)
    al.alSourcePlay(source_id)

# Cargar sonidos
pong_sound = load_sound("pong.wav")
win_sound = load_sound("win.wav")

def get_hand_rect(hand_landmarks):
    x_coords = [lm.x * WIDTH for lm in hand_landmarks.landmark]
    y_coords = [lm.y * HEIGHT for lm in hand_landmarks.landmark]
    return (int(min(x_coords)), int(min(y_coords))), (int(max(x_coords)), int(max(y_coords)))

def process_hands(results):
    global left_paddle_y, right_paddle_y
    
    if not results.multi_hand_landmarks:
        return []

    hands_sorted = sorted(
        zip(results.multi_hand_landmarks, results.multi_handedness),
        key=lambda h: h[0].landmark[mp_hands.HandLandmark.WRIST].x
    )

    hand_data = []
    for idx, (landmarks, handedness) in enumerate(hands_sorted[:2], 1):
        rect = get_hand_rect(landmarks)
        center_y = (rect[0][1] + rect[1][1]) // 2
        
        if idx == 1:
            left_paddle_y = max(0, min(HEIGHT - RECTANGULO_HEIGHT, center_y - RECTANGULO_HEIGHT // 2))
        else:
            right_paddle_y = max(0, min(HEIGHT - RECTANGULO_HEIGHT, center_y - RECTANGULO_HEIGHT // 2))
        
        hand_data.append((idx, rect))
    
    return hand_data

def update_ball_position():
    global ballPosition, ballSpeedX, ballSpeedY, left_score, right_score, last_touched
    
    ballPosition[0] += ballSpeedX
    ballPosition[1] += ballSpeedY
    
    # Rebote en bordes superior e inferior
    if ballPosition[1] <= 0 or ballPosition[1] >= HEIGHT:
        ballSpeedY = -ballSpeedY
    
    # Colisión con paletas
    if (POS_HORIZONTAL_IZQUIERDA - 15 <= ballPosition[0] <= POS_HORIZONTAL_IZQUIERDA + 15 and
        left_paddle_y <= ballPosition[1] <= left_paddle_y + RECTANGULO_HEIGHT):
        if last_touched != 1:
            ballSpeedX = abs(ballSpeedX)
            last_touched = 1
            play_sound(pong_sound, pan=-3.0)
    
    elif (POS_HORIZONTAL_DERECHA - 15 <= ballPosition[0] <= POS_HORIZONTAL_DERECHA + 15 and
          right_paddle_y <= ballPosition[1] <= right_paddle_y + RECTANGULO_HEIGHT):
        if last_touched != 2:
            ballSpeedX = -abs(ballSpeedX)
            last_touched = 2
            play_sound(pong_sound, pan=3.0)
    
    # Puntos
    if ballPosition[0] <= 0:
        right_score += 1
        play_sound(win_sound, pan=5.0)
        reset_ball()
    elif ballPosition[0] >= WIDTH:
        left_score += 1
        play_sound(win_sound, pan=-5.0)
        reset_ball()

def reset_ball():
    global ballPosition, last_touched
    ballPosition = [WIDTH // 2, HEIGHT // 2]
    last_touched = None

def draw_interface(frame):
    # Línea central
    for y in range(0, HEIGHT, 20):
        if (y // 10) % 2 == 0:
            cv2.line(frame, (WIDTH//2, y), (WIDTH//2, y+10), (255,255,255), 2)
    
    # Paletas
    cv2.rectangle(frame, (POS_HORIZONTAL_IZQUIERDA-15, left_paddle_y),
                 (POS_HORIZONTAL_IZQUIERDA+15, left_paddle_y+RECTANGULO_HEIGHT),
                 (255,255,255), -1)
    cv2.rectangle(frame, (POS_HORIZONTAL_DERECHA-15, right_paddle_y),
                 (POS_HORIZONTAL_DERECHA+15, right_paddle_y+RECTANGULO_HEIGHT),
                 (255,255,255), -1)
    
    # Bola
    cv2.circle(frame, tuple(ballPosition), BALL_SIZE, (255,255,255), -1)
    
    # Marcador
    cv2.putText(frame, f"Left: {left_score}", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f"Right: {right_score}", (WIDTH-200, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

def handle_countdown(frame, hand_data):
    global ballSpeedX, ballSpeedY
    
    if len(hand_data) < 2:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (WIDTH, HEIGHT), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        text = f"Jugadores detectados: {len(hand_data)}/2"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.putText(frame, text, ((WIDTH-text_size[0])//2, (HEIGHT+text_size[1])//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        ballSpeedX = ballSpeedY = 0
        return False
    else:
        if ballSpeedX == 0 and ballSpeedY == 0:
            for i in range(COUNT_DOWN, 0, -1):
                temp_frame = frame.copy()
                overlay = temp_frame.copy()
                cv2.rectangle(overlay, (0,0), (WIDTH, HEIGHT), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.7, temp_frame, 0.3, 0, temp_frame)
                
                cv2.putText(temp_frame, str(i), (WIDTH//2-30, HEIGHT//2+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
                cv2.imshow("Pong AR", temp_frame)
                cv2.waitKey(1000)
            
            ballSpeedX = 5 if last_touched == 2 else -5
            ballSpeedY = 5
        return True

cap = cv2.VideoCapture(0)