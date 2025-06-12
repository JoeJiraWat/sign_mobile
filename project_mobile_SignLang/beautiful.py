import cv2
import mediapipe as mp
import time
import numpy as np

# Initial setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def detect():
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    face_mesh = mp_face.FaceMesh()
    cap = cv2.VideoCapture(0)

    phase = 0
    loop_path = []
    loop_start_time = None
    thumb_up_start = None
    start_time = None

    success = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(rgb)
        face_results = face_mesh.process(rgb)

        face_center = None
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_draw.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_CONTOURS)
                nose = face_landmarks.landmark[1]  # Nose tip
                face_center = (int(nose.x * w), int(nose.y * h))

        if hand_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                label = handedness.classification[0].label
                if label != "Right":
                    continue

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))

                # Phase 0: มือขวาใกล้หน้า
                if phase == 0 and face_center and distance(index_pos, face_center) < 100:
                    phase = 1
                    loop_path = []
                    loop_start_time = time.time()
                    print("Phase 1: เริ่มลูบหน้า")

                # Phase 1: ลูบหน้ารอบเดียว
                elif phase == 1:
                    loop_path.append(index_pos)
                    if time.time() - loop_start_time > 2.0:
                        phase = 2
                        thumb_up_start = None
                        print("Phase 2: ยกนิ้วโป้ง")

                # Phase 2: ตรวจนิ้วโป้ง
                elif phase == 2:
                    dy = index_pos[1] - thumb_pos[1]
                    dx = abs(index_pos[0] - thumb_pos[0])
                    if dy > 40 and dx < 50:
                        if thumb_up_start is None:
                            thumb_up_start = time.time()
                        elif time.time() - thumb_up_start > 0.5:
                            end_time = time.time()
                            duration = end_time - loop_start_time
                            print("✅ ท่าสวยสำเร็จ!")
                            print(f"🕒 ใช้เวลา: {duration:.2f} วินาที")
                            cv2.putText(frame, f"Success! Time: {duration:.2f}s",
                                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                            cv2.imshow("Pose: Beautiful", frame)
                            cv2.waitKey(3000)
                            success = True
                            cap.release()
                            cv2.destroyAllWindows()
                            return success
                    else:
                        thumb_up_start = None

        cv2.putText(frame, f"Phase: {phase}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Pose: Beautiful", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return success
