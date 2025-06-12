# /Users/b4byf4lc0n/project_mobile_SignLang/gesture_processors.py
import cv2
import mediapipe as mp
import numpy as np
import time

class BaseGestureProcessor:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles # For consistency if needed
        self.reset()

    def process_frame(self, bgr_frame_input, frame_width, frame_height):
        """
        Processes a single frame to detect a gesture.
        Input frame should be BGR and already horizontally flipped if necessary.
        Returns:
            - processed_bgr_frame: Frame with drawings (BGR format).
            - gesture_recognized: Boolean, True if gesture is complete.
            - status_text: String, current status/phase of detection.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def reset(self):
        """Resets the internal state of the gesture processor."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

class BeautifulGestureProcessor(BaseGestureProcessor):
    def __init__(self):
        super().__init__()
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.hands_detector = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) # Focus on one hand
        self.face_mesh_detector = self.mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
        self.reset()

    def reset(self):
        self.phase = 0
        self.loop_path = []
        self.loop_start_time = None
        self.thumb_up_start_time = None
        self.gesture_completed_flag = False

    def process_frame(self, bgr_frame_input, frame_width, frame_height):
        if self.gesture_completed_flag:
            return bgr_frame_input, True, "ท่าสวยสำเร็จ!"

        # Frame is already flipped horizontally by KivyCamera
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(bgr_frame_input, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        hand_results = self.hands_detector.process(rgb_frame)
        face_results = self.face_mesh_detector.process(rgb_frame)

        rgb_frame.flags.writeable = True
        output_bgr_frame = bgr_frame_input.copy() # Draw on a copy

        face_center = None
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # self.mp_drawing.draw_landmarks(
                #     output_bgr_frame, face_landmarks, self.mp_face.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None, # No landmarks
                #     connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                # )
                nose_tip = face_landmarks.landmark[1] # Nose tip landmark index
                face_center = (int(nose_tip.x * frame_width), int(nose_tip.y * frame_height))

        if hand_results.multi_hand_landmarks:
            # Assuming we are interested in the 'Right' hand from the user's perspective
            # (which will appear as 'Left' if the image is flipped, or if MediaPipe reports based on image orientation)
            # For simplicity, we'll process the first detected hand if it's likely the dominant one.
            # The original script checked for "Right" handedness. If the frame is flipped, this might become "Left".
            # Let's assume the user uses their dominant hand, and we process the first one detected.
            for hand_landmarks in hand_results.multi_hand_landmarks: # Iterate through detected hands
                # handedness = hand_results.multi_handedness[hand_idx].classification[0].label
                # if handedness != "Right": # Original script logic
                #     continue

                self.mp_drawing.draw_landmarks(
                    output_bgr_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_pos = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))
                thumb_pos = (int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height))

                if self.phase == 0: # มือขวาใกล้หน้า
                    if face_center and self._distance(index_pos, face_center) < 100: # Adjusted threshold
                        self.phase = 1
                        self.loop_path = []
                        self.loop_start_time = time.time()
                        # print("Beautiful Phase 1: Start face loop")
                elif self.phase == 1: # ลูบหน้ารอบเดียว
                    self.loop_path.append(index_pos)
                    if time.time() - self.loop_start_time > 1.5: # Adjusted time for loop
                        self.phase = 2
                        self.thumb_up_start_time = None
                        # print("Beautiful Phase 2: Thumb up check")
                elif self.phase == 2: # ตรวจนิ้วโป้ง
                    dy = index_pos[1] - thumb_pos[1] # Thumb tip Y should be lower (higher value) than index tip Y
                    dx = abs(index_pos[0] - thumb_pos[0])
                    # Check if thumb is below index and relatively aligned vertically
                    if thumb_pos[1] > index_tip.y * frame_height + 20 and dy > 30 and dx < 60 : # Thumb is up (relative to palm)
                        if self.thumb_up_start_time is None:
                            self.thumb_up_start_time = time.time()
                        elif time.time() - self.thumb_up_start_time > 0.3: # Hold for 0.3s
                            self.gesture_completed_flag = True
                            # print("✅ ท่าสวยสำเร็จ!")
                            return output_bgr_frame, True, "ท่าสวยสำเร็จ!"
                    else:
                        self.thumb_up_start_time = None # Reset if thumb is not up

        status_text = f"สวย: เฟส {self.phase}"
        return output_bgr_frame, False, status_text

class WhereGestureProcessor(BaseGestureProcessor):
    def __init__(self):
        super().__init__()
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands_detector = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.pose_detector = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.reset()

    def reset(self):
        self.phase = 0
        self.overall_start_time = None # Time when processing for this gesture attempt starts
        self.sway_start_time = None
        self.prev_x_right_hand = None
        self.gesture_completed_flag = False

    def process_frame(self, bgr_frame_input, frame_width, frame_height):
        if self.gesture_completed_flag:
            return bgr_frame_input, True, "ท่าที่ไหนสำเร็จ!"

        if self.overall_start_time is None:
            self.overall_start_time = time.time()

        rgb_frame = cv2.cvtColor(bgr_frame_input, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        hand_results = self.hands_detector.process(rgb_frame)
        pose_results = self.pose_detector.process(rgb_frame)

        rgb_frame.flags.writeable = True
        output_bgr_frame = bgr_frame_input.copy()

        chest_y = None
        if pose_results.pose_landmarks:
            # self.mp_drawing.draw_landmarks(
            #     output_bgr_frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            # )
            l_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            if l_shoulder.visibility > 0.5 and r_shoulder.visibility > 0.5: # Check visibility
                chest_y = int(((l_shoulder.y + r_shoulder.y) / 2) * frame_height)
                # cv2.line(output_bgr_frame, (0, chest_y), (frame_width, chest_y), (100, 100, 255), 2)


        if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
            hand_coords_tips = []
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(
                    output_bgr_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                     self.mp_drawing_styles.get_default_hand_landmarks_style(),
                     self.mp_drawing_styles.get_default_hand_connections_style()
                )
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cx = int(index_finger_tip.x * frame_width)
                cy = int(index_finger_tip.y * frame_height)
                hand_coords_tips.append({'x': cx, 'y': cy, 'handedness': hand_results.multi_handedness[hand_idx].classification[0].label})

            # Sort hands by x-coordinate to identify left and right hand on screen
            hand_coords_tips.sort(key=lambda h: h['x'])
            left_hand_tip = hand_coords_tips[0]
            right_hand_tip = hand_coords_tips[1]

            if self.phase == 0: # Both hands above chest level, forming 'T' shape (index fingers pointing up)
                # This phase needs more specific finger orientation checks from original script if they existed.
                # For now, just check position relative to chest.
                if chest_y and left_hand_tip['y'] < chest_y and right_hand_tip['y'] < chest_y:
                     # Check if index fingers are pointing up (simplified: y of MCP is below y of TIP)
                    left_mcp_y = hand_results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y * frame_height
                    right_mcp_y = hand_results.multi_hand_landmarks[1].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y * frame_height
                    if left_hand_tip['y'] < left_mcp_y and right_hand_tip['y'] < right_mcp_y:
                        self.phase = 1
                        # print("Where Phase 1: Hands up, T-pose like")
            elif self.phase == 1: # Hands close together (index tips touching)
                vertical_dist = abs(left_hand_tip['y'] - right_hand_tip['y'])
                horizontal_dist = abs(left_hand_tip['x'] - right_hand_tip['x'])
                if vertical_dist < 50 and horizontal_dist < 50: # Increased threshold
                    self.phase = 2
                    self.sway_start_time = None
                    self.prev_x_right_hand = None
                    # print("Where Phase 2: Hands together, start sway check")
            elif self.phase == 2: # Swaying motion
                movement_detected = False
                current_x_right_hand = right_hand_tip['x']
                if self.prev_x_right_hand is not None:
                    dx = abs(current_x_right_hand - self.prev_x_right_hand)
                    if dx > 15: # Adjusted threshold for movement
                        movement_detected = True

                if movement_detected:
                    if self.sway_start_time is None:
                        self.sway_start_time = time.time()
                        # print("Sway detected, timer started.")
                    elif time.time() - self.sway_start_time >= 0.8: # Sway for 0.8 seconds
                        self.gesture_completed_flag = True
                        # print("✅ ท่าที่ไหนสำเร็จ")
                        return output_bgr_frame, True, "ท่าที่ไหนสำเร็จ!"
                else:
                    # If no movement for a short while, reset sway timer
                    if self.sway_start_time and (time.time() - self.sway_start_time > 0.5): # Reset if no movement for 0.5s
                        self.sway_start_time = None

                self.prev_x_right_hand = current_x_right_hand
        
        status_text = f"ที่ไหน: เฟส {self.phase}"
        if self.phase == 2 and self.sway_start_time:
            status_text += f" (กำลังส่าย: {time.time() - self.sway_start_time:.1f}s)"

        return output_bgr_frame, False, status_text

