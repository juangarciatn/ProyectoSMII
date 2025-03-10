import cv2
import mediapipe as mp
import numpy as np

# Configuración de la ventana
WIDTH = 640
HEIGHT = 480

# Tamaño de la bola
BALL_SIZE = 10

# Velocidad de la bola
ballSpeedX = 5
ballSpeedY = 5

# Posición inicial de la bola
ballPosition = [WIDTH // 2, HEIGHT // 2]

# Puntuación
left_score = 0
right_score = 0

# Variable para evitar múltiples golpes consecutivos
ball_collided = False

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils

# Variable global para almacenar los rectángulos de las manos
hand_rects = []

# Función para obtener el rectángulo que engloba la mano
def get_hand_rect(hand_landmarks):
    x_coords = [lm.x * WIDTH for lm in hand_landmarks.landmark]
    y_coords = [lm.y * HEIGHT for lm in hand_landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    return (min_x, min_y), (max_x, max_y)  # Esquina superior izquierda y esquina inferior derecha

# Función para dibujar la bola
def draw_ball(frame):
    cv2.circle(frame, tuple(ballPosition), BALL_SIZE, (255, 255, 255), -1)

# Función para actualizar la posición de la bola
def update_ball_position(hand_rects):
    global ballPosition, ballSpeedX, ballSpeedY, left_score, right_score, ball_collided

    # Mover la bola
    ballPosition[0] += ballSpeedX
    ballPosition[1] += ballSpeedY

    # Rebotar en los bordes superior e inferior
    if ballPosition[1] <= 0 or ballPosition[1] >= HEIGHT:
        ballSpeedY = -ballSpeedY

    # Rebotar en las manos (rectángulos)
    for i, rect in enumerate(hand_rects):
        (min_x, min_y), (max_x, max_y) = rect
        if (min_x <= ballPosition[0] <= max_x) and (min_y <= ballPosition[1] <= max_y):
            if not ball_collided:  # Solo rebotar si no ha colisionado recientemente
                ballSpeedX = -ballSpeedX
                ball_collided = True  # Marcar que la bola ha colisionado
                # Incrementar la puntuación correspondiente
                if i == 0:  # Primera mano (izquierda)
                    left_score += 1
                elif i == 1:  # Segunda mano (derecha)
                    right_score += 1
            break
        else:
            ball_collided = False  # Restablecer si la bola no está en contacto

    # Reiniciar la bola si sale por los lados
    if ballPosition[0] <= 0 or ballPosition[0] >= WIDTH:
        ballPosition = [WIDTH // 2, HEIGHT // 2]
        ball_collided = False  # Restablecer la colisión al reiniciar la bola

# Función para mostrar la puntuación
def draw_score(frame):
    cv2.putText(frame, f"Left: {left_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Right: {right_score}", (WIDTH - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Inicializar la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Añadir un contador de frames y un intervalo de detección
frame_counter = 0
detection_interval = 2  # Detectar manos cada 2 frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear el frame horizontalmente para que sea como un espejo
    frame = cv2.flip(frame, 1)

    # Convertir la imagen a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la detección de manos solo cada N frames
    if frame_counter % detection_interval == 0:
        results = hands.process(rgb_frame)

        # Limpiar la lista de rectángulos
        hand_rects.clear()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Obtener el rectángulo que engloba la mano
                rect = get_hand_rect(hand_landmarks)
                hand_rects.append(rect)

    # Dibujar los rectángulos en cada frame
    for rect in hand_rects:
        cv2.rectangle(frame, rect[0], rect[1], (0, 255, 0), 2)

    # Incrementar el contador de frames
    frame_counter += 1

    # Dibujar la bola
    draw_ball(frame)

    # Actualizar la posición de la bola
    update_ball_position(hand_rects)

    # Mostrar la puntuación
    draw_score(frame)

    # Mostrar el frame
    cv2.imshow("Pong AR", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
