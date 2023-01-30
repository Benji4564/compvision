import cv2
import mediapipe as mp
import time
import math

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            max_distance = 0
            finger_tip = None
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                if id > 1:
                    # Calculate the distance between the first and current landmark
                    distance = ((cx - handLms.landmark[0].x * w) ** 2 + (cy - handLms.landmark[0].y * h) ** 2) ** 0.5
                    if distance > max_distance:
                        max_distance = distance
                        finger_tip = (cx, cy)
            if finger_tip:
                # Draw a line from the first landmark to the finger tip
                cv2.line(img, (int(handLms.landmark[0].x * w), int(handLms.landmark[0].y * h)), finger_tip, (0, 255, 0), 5)
                x_diff = finger_tip[0] - handLms.landmark[0].x * w
                y_diff = finger_tip[1] - handLms.landmark[0].y * h
                angle = math.degrees(math.atan2(y_diff, x_diff))
                if angle < 0:
                    angle += 360
                direction = None
                if (angle >= 337.5 or angle < 22.5):
                    direction = "Right"
                elif (angle >= 22.5 and angle < 67.5):
                    direction = "Up Right"
                elif (angle >= 67.5 and angle < 112.5):
                    direction = "Up"
                elif (angle >= 112.5 and angle < 157.5):
                    direction = "Up Left"
                elif (angle >= 157.5 and angle < 202.5):
                    direction = "Left"
                elif (angle >= 202.5 and angle < 247.5):
                    direction = "Down Left"
                elif (angle >= 247.5 and angle < 292.5):
                    direction = "Down"
                elif (angle >= 292.5 and angle < 337.5):
                    direction = "Down Right"
                print(direction)
