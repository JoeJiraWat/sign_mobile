from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.video import Video
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.anchorlayout import AnchorLayout

import cv2
import mediapipe as mp
import time

Window.clearcolor = (1, 1, 1, 1)

Builder.load_string('''
<MainScreen>:
    FloatLayout:
        canvas.before:
            Color:
                rgba: 1, 1, 1, 1
            Rectangle:
                pos: self.pos
                size: self.size

        Image:
            source: 'logo_app.png'
            size_hint: 0.6, 0.6
            pos_hint: {'center_x': 0.5, 'center_y': 0.6}

        Button:
            text: 'Start Learning'
            font_size: 22
            size_hint: 0.4, 0.12
            pos_hint: {'center_x': 0.5, 'y': 0.1}
            background_normal: ''
            background_color: 0.1, 0.2, 0.6, 1  # น้ำเงินเข้ม
            border: [20, 20, 20, 20]
            on_release:
                app.root.current = 'vocab'

<VocabScreen>:
    FloatLayout:
        canvas.before:
            Color:
                rgba: 1, 1, 1, 1
            Rectangle:
                pos: self.pos
                size: self.size

        Button:
            text: '<'
            font_size: 50
            size_hint: 0.18, 0.08
            pos_hint: {'x': 0.02, 'top': 0.98}
            background_normal: ''
            background_color: 1, 1, 1, 1  # ขาว
            color: 0, 0, 0, 1
            border: [20, 20, 20, 20]
            on_release:
                app.root.current = 'main'

        Label:
            text: 'Choose Vocabulary'
            font_size: 30
            color: 0, 0, 0.2, 1
            size_hint: None, None
            size: self.texture_size
            pos_hint: {'center_x': 0.5, 'top': 0.88}

        Button:
            text: 'Where'
            font_size: 22
            size_hint: 0.4, 0.12
            pos_hint: {'center_x': 0.5, 'center_y': 0.55}
            background_normal: ''
            background_color: 0.1, 0.2, 0.6, 1
            border: [20, 20, 20, 20]
            on_release:
                root.manager.get_screen('video').word = 'where'
                root.manager.current = 'video'

        Button:
            text: 'Beautiful'
            font_size: 22
            size_hint: 0.4, 0.12
            pos_hint: {'center_x': 0.5, 'center_y': 0.38}
            background_normal: ''
            background_color: 0.1, 0.2, 0.6, 1
            border: [20, 20, 20, 20]
            on_release:
                root.manager.get_screen('video').word = 'beautiful'
                root.manager.current = 'video'

<VideoScreen>:
    FloatLayout:
        canvas.before:
            Color:
                rgba: 1, 1, 1, 1
            Rectangle:
                pos: self.pos
                size: self.size

        Button:
            text: '<'
            font_size: 50
            size_hint: 0.18, 0.08
            pos_hint: {'x': 0.02, 'top': 0.98}
            background_normal: ''
            background_color: 1, 1, 1, 1
            color: 0, 0, 0, 1
            border: [20, 20, 20, 20]
            on_release:
                app.root.current = 'vocab'

        Video:
            id: vid
            source: ''
            state: 'play'
            options: {'eos': 'loop'}
            allow_stretch: True
            size_hint: 0.95, 0.5
            pos_hint: {'center_x': 0.5, 'top': 0.88}

        Label:
            id: pass_label
            text: ''
            font_size: 40
            bold: True
            color: 0, 1, 0, 1
            size_hint: None, None
            size: self.texture_size
            pos_hint: {'center_x': 0.5, 'center_y': 0.5}

        Image:
            id: cam
            allow_stretch: True
            keep_ratio: True
            size_hint: 0.95, 0.35
            pos_hint: {'center_x': 0.5, 'y': 0.12}

        BoxLayout:
            orientation: 'horizontal'
            spacing: 10
            size_hint: 0.85, 0.08
            pos_hint: {'center_x': 0.5, 'y': 0.02}

            Button:
                text: 'Start Detection'
                font_size: 18
                background_normal: ''
                background_color: 0.1, 0.2, 0.6, 1
                border: [20, 20, 20, 20]
                on_release:
                    root.start_detection()

            Button:
                text: 'Replay'
                font_size: 18
                background_normal: ''
                background_color: 0.1, 0.2, 0.6, 1
                border: [20, 20, 20, 20]
                on_release:
                    root.replay_video()
''')

class MainScreen(Screen):
    pass

class VocabScreen(Screen):
    pass

class VideoScreen(Screen):
    word = ''
    detecting = False


    def on_enter(self):
        self.ids.vid.source = f'{self.word}.mp4'
        self.ids.vid.state = 'play'
        self.ids.pass_label.text = ''
        self.detecting = False

    def replay_video(self):
        self.ids.vid.state = 'stop'
        self.ids.vid.position = 0
        self.ids.vid.state = 'play'

    def start_detection(self):
        if not self.detecting:
            self.ids.vid.state = 'stop'
            self.detecting = True
            self.ids.pass_label.text = ''
            self.capture = cv2.VideoCapture(0)
            self.success = False
            self.phase = 0
            self.prev_x = None
            self.sway_start_time = None
            self.start_time = None
            self.hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
            self.pose = mp.solutions.pose.Pose()
            Clock.schedule_interval(self.update, 1.0 / 30)

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_result = self.hands.process(rgb)
        pose_result = self.pose.process(rgb)

        if self.start_time is None:
            self.start_time = time.time()

        chest_y = None
        if pose_result.pose_landmarks:
            l_shoulder = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            chest_y = int(((l_shoulder.y + r_shoulder.y) / 2) * h)

        if hands_result.multi_hand_landmarks and len(hands_result.multi_hand_landmarks) == 2:
            coords = []
            for handLms in hands_result.multi_hand_landmarks:
                index_tip = handLms.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                coords.append((cx, cy))
            coords.sort(key=lambda x: x[0])
            left_index, right_index = coords

            if self.phase == 0:
                if chest_y and left_index[1] < chest_y and right_index[1] < chest_y:
                    self.phase = 1
            elif self.phase == 1:
                vertical_dist = abs(left_index[1] - right_index[1])
                horizontal_dist = abs(left_index[0] - right_index[0])
                if vertical_dist < 40 and horizontal_dist < 40:
                    self.phase = 2
                    self.sway_start_time = None
                    self.prev_x = None
            elif self.phase == 2:
                movement_detected = False
                if self.prev_x is not None:
                    dx = abs(self.prev_x - right_index[0])
                    if dx > 10:
                        movement_detected = True

                if movement_detected:
                    if self.sway_start_time is None:
                        self.sway_start_time = time.time()
                    elif time.time() - self.sway_start_time >= 1.0:
                        self.success = True
                        self.stop_detection()
                else:
                    self.sway_start_time = None

                self.prev_x = right_index[0]

        # Show camera
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.cam.texture = texture

    def stop_detection(self):
        Clock.unschedule(self.update)
        self.capture.release()
        self.detecting = False
        self.ids.cam.texture = None
        self.ids.pass_label.text = "✅ PASSED"
        print("✅ Passed!")
        Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'main'), 2)

class SignApp(App):
    def build(self):
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(VocabScreen(name='vocab'))
        sm.add_widget(VideoScreen(name='video'))
        return sm

if __name__ == '__main__':
    SignApp().run()
