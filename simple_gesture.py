import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import keyboard
import time

pyautogui.FAILSAFE = False

# ---------- Screen Setup ----------
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# ---------- MediaPipe Setup ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ---------- Mouse Smoothing ----------
prev_mouse_x, prev_mouse_y = screen_w // 2, screen_h // 2
SMOOTHING = 5

# ---------- Flags ----------
click_done = False
drag_active = False
minimize_done = False
close_done = False

# Double click detection
last_pinch_time = 0
pinch_count = 0

def get_finger_status(lm_list):
    fingers = []
    # Thumb (Right hand)
    if lm_list[4][0] > lm_list[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)
    # Index, Middle, Ring, Pinky
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        if lm_list[tip][1] < lm_list[pip][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# ---------- Main Loop ----------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    gesture_text = "Waiting for hand... 🖐️"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if len(lm_list) != 0:
                fingers = get_finger_status(lm_list)
                f_thumb, f_index, f_middle, f_ring, f_pinky = fingers

                x4, y4 = lm_list[4]   # Thumb tip
                x8, y8 = lm_list[8]   # Index tip
                x12, y12 = lm_list[12] # Middle tip
                x16, y16 = lm_list[16] # Ring tip

                # Distances for pinch gestures
                dist_index = math.hypot(x8 - x4, y8 - y4)    # Thumb + Index
                dist_middle = math.hypot(x12 - x4, y12 - y4) # Thumb + Middle
                dist_ring = math.hypot(x16 - x4, y16 - y4)   # Thumb + Ring

                # =========================================================
                # 1. MOUSE MOVE (Only Index Finger UP, all others DOWN)
                # =========================================================
                if f_index == 1 and f_middle == 0 and f_ring == 0 and f_pinky == 0:
                    target_x = np.interp(x8, [50, w - 50], [0, screen_w])
                    target_y = np.interp(y8, [50, h - 50], [0, screen_h])
                    mouse_x = prev_mouse_x + (target_x - prev_mouse_x) / SMOOTHING
                    mouse_y = prev_mouse_y + (target_y - prev_mouse_y) / SMOOTHING
                    pyautogui.moveTo(mouse_x, mouse_y)
                    prev_mouse_x, prev_mouse_y = mouse_x, mouse_y
                    gesture_text = "MOUSE MOVE 🖱️"

                # =========================================================
                # 2. LEFT CLICK (Thumb + Index Pinch)
                # =========================================================
                if dist_index < 30 and f_index == 1:
                    if not click_done:
                        pyautogui.click()
                        click_done = True
                        gesture_text = "LEFT CLICK ✅"
                else:
                    click_done = False

                # =========================================================
                # 3. DRAG & DROP (Thumb + Middle Pinch)
                # =========================================================
                if dist_middle < 30 and f_middle == 1:
                    if not drag_active:
                        pyautogui.mouseDown()
                        drag_active = True
                        gesture_text = "DRAG START 📦"
                    else:
                        # Move hand while dragging
                        cx_hand = int(np.mean([lm_list[i][0] for i in range(21)]))
                        cy_hand = int(np.mean([lm_list[i][1] for i in range(21)]))
                        mouse_x = np.interp(cx_hand, [50, w - 50], [0, screen_w])
                        mouse_y = np.interp(cy_hand, [50, h - 50], [0, screen_h])
                        pyautogui.moveTo(mouse_x, mouse_y, duration=0.05)
                        gesture_text = "DRAGGING... 📦"
                else:
                    if drag_active:
                        pyautogui.mouseUp()
                        drag_active = False
                        gesture_text = "DROP ✅"

                # =========================================================
                # 4. MINIMIZE WINDOW (Thumb + Ring Pinch)
                # =========================================================
                if dist_ring < 30 and f_ring == 1:
                    if not minimize_done:
                        keyboard.press_and_release('win+down')
                        minimize_done = True
                        gesture_text = "MINIMIZE WINDOW "
                else:
                    minimize_done = False

                # =========================================================
                # 5. CLOSE WINDOW (Double Click: Pinch Thumb+Index twice quickly)
                # =========================================================
                current_time = time.time()
                
                if dist_index < 30 and f_index == 1:
                    if not close_done:
                        # Count pinch events
                        if current_time - last_pinch_time < 0.5:  # within 0.5 seconds
                            pinch_count += 1
                        else:
                            pinch_count = 1
                        last_pinch_time = current_time
                        
                        if pinch_count >= 2:
                            keyboard.press_and_release('alt+f4')
                            close_done = True
                            pinch_count = 0
                            gesture_text = "CLOSE WINDOW ❌"
                else:
                    # Reset if hand opens
                    if not (dist_index < 30 and f_index == 1):
                        close_done = False

    # ---------- Display on Screen ----------
    cv2.rectangle(frame, (10, 10), (600, 80), (0, 0, 0), -1)
    cv2.putText(frame, gesture_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # ---------- Legend / Instructions ----------
    cv2.rectangle(frame, (10, h - 150), (600, h - 10), (0, 0, 0), -1)
    cv2.putText(frame, "Index = Move  |  Thumb+Index = Click  |  Thumb+Middle = Drag", 
                (20, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    cv2.putText(frame, "Thumb+Ring = Minimize  |  Double Click (Index+Thumb) = Close", 
                (20, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    cv2.putText(frame, "Press 'Q' to Quit", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Simple Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()