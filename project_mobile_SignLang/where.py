#where
import cv2
import mediapipe as mp
import numpy as np
import time

Init mediapipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)
start_time = None
phase = 0
sway_start_time = None
prev_x = None

def getdistance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w,  = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hands_result = hands.process(rgb)
    pose_result = pose.process(rgb)

    if start_time is None:
        start_time = time.time()

    # Chest level line
    chest_y = None
    if pose_result.pose_landmarks:
        l_shoulder = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        chest_y = int(((l_shoulder.y + r_shoulder.y) / 2) * h)
        cv2.line(frame, (0, chest_y), (w, chest_y), (100, 100, 255), 2)

    if hands_result.multi_hand_landmarks and len(hands_result.multi_hand_landmarks) == 2:
        coords = []
        for handLms in hands_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            index_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            coords.append((cx, cy))

        coords.sort(key=lambda x: x[0])  # Left hand on the left side
        left_index, right_index = coords

        if phase == 0:
            if chest_y and left_index[1] < chest_y and right_index[1] < chest_y:
                phase = 1
                print("Phase 1: ท่านิ้ว T")
        elif phase == 1:
            vertical_dist = abs(left_index[1] - right_index[1])
            horizontal_dist = abs(left_index[0] - right_index[0])
            if vertical_dist < 40 and horizontal_dist < 40:
                phase = 2
                sway_start_time = None
                prev_x = None
                print("Phase 2: เริ่มจับเวลาส่ายมือ")

        elif phase == 2:
            movement_detected = False
            if prev_x is not None:
                dx = abs(prev_x - right_index[0])
                if dx > 10:
                    movement_detected = True

            if movement_detected:
                if sway_start_time is None:
                    sway_start_time = time.time()
                    print("เริ่มจับเวลาส่ายมือ...")
                elif time.time() - sway_start_time >= 1.0:
                    total_duration = time.time() - start_time
                    print("✅ ท่าที่ไหนสำเร็จ")
                    print(f"🕒 เวลาในการทำท่า: {total_duration:.2f} วินาที")
                    cv2.putText(frame, f"Success! Time: {total_duration:.2f}s",
                                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.imshow("Where Sign Detection", frame)
                    cv2.waitKey(3000)
                    break
            else:
                sway_start_time = None  # รีเซ็ตถ้าหยุดเคลื่อนไหว

            prev_x = right_index[0]

    cv2.putText(frame, f"Phase: {phase}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Where Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
