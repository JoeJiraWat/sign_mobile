import os
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.uix.video import Video
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import BooleanProperty, StringProperty, ObjectProperty

import cv2
import numpy as np

# MediaPipe components will be imported conditionally in KivyCamera
mp_hands = None
mp_drawing = None
mp_drawing_styles = None

# KV Language String
KV_STRING = """
#:import FadeTransition kivy.uix.screenmanager.FadeTransition

<BaseScreen@Screen>: # Base screen for consistent font if needed
    font_name: 'Garuda' # Default font for Thai text

<MainScreen(BaseScreen)>:
    BoxLayout:
        orientation: 'vertical'
        padding: dp(50)
        spacing: dp(20)
        Image:
            source: 'logo.png' 
            allow_stretch: True
            keep_ratio: True
            size_hint_y: 0.7
        Button:
            text: 'เริ่มต้นใช้งาน'
            font_name: root.font_name
            font_size: '20sp'
            size_hint_y: 0.3
            on_press: app.root.current = 'vocab_menu'

<VocabMenuScreen(BaseScreen)>:
    BoxLayout:
        orientation: 'vertical'
        padding: dp(50)
        spacing: dp(20)
        Label:
            text: 'เมนูหลัก'
            font_name: root.font_name
            font_size: '30sp'
            size_hint_y: 0.2
        Button:
            text: 'คำศัพท์'
            font_name: root.font_name
            font_size: '20sp'
            size_hint_y: 0.4
            on_press: app.root.current = 'word_selection'
        Button:
            text: 'กลับหน้าแรก'
            font_name: root.font_name
            font_size: '18sp'
            size_hint_y: 0.2
            on_press: app.root.current = 'main'


<WordSelectionScreen(BaseScreen)>:
    BoxLayout:
        orientation: 'vertical'
        padding: dp(30)
        spacing: dp(15)
        Label:
            text: 'เลือกคำศัพท์'
            font_name: root.font_name
            font_size: '30sp'
            size_hint_y: 0.2
        Button:
            text: 'ที่ไหน'
            font_name: root.font_name
            font_size: '20sp'
            size_hint_y: 0.3
            on_press: app.select_word('ที่ไหน')
        Button:
            text: 'สวย'
            font_name: root.font_name
            font_size: '20sp'
            size_hint_y: 0.3
            on_press: app.select_word('สวย')
        Button:
            text: 'กลับเมนูหลัก'
            font_name: root.font_name
            font_size: '18sp'
            size_hint_y: 0.15
            on_press: app.root.current = 'vocab_menu'

<LearningScreen(BaseScreen)>:
    video_player: video_player
    kivy_camera: kivy_camera
    status_label: status_label
    next_button: next_button
    back_to_menu_button: back_to_menu_button

    BoxLayout:
        orientation: 'vertical'
        padding: dp(10)
        spacing: dp(10)

        Video:
            id: video_player
            source: root.video_source
            state: 'stop'
            options: {'eos': 'stop'} 
            allow_stretch: True
            size_hint_y: 0.6 if root.show_video_player else 0
            opacity: 1 if root.show_video_player else 0
            on_eos: root.on_video_end()

        KivyCamera:
            id: kivy_camera
            size_hint_y: 0.6 if root.show_camera_view else 0
            opacity: 1 if root.show_camera_view else 0
            on_sign_detected: root.on_sign_recognized_event() # Connect event

        Label:
            id: status_label
            text: root.status_message
            font_name: root.font_name
            font_size: '24sp'
            size_hint_y: 0.1 if root.status_message else 0
            opacity: 1 if root.status_message else 0
            halign: 'center'

        Button:
            id: next_button
            text: 'ถัดไป'
            font_name: root.font_name
            font_size: '20sp'
            on_press: root.prepare_for_camera()
            size_hint_y: 0.15
            opacity: 1 if root.show_next_button else 0
            disabled: not root.show_next_button

        Button:
            id: back_to_menu_button
            text: 'กลับไปหน้าคำศัพท์'
            font_name: root.font_name
            font_size: '20sp'
            on_press: root.go_to_word_selection()
            size_hint_y: 0.15
            opacity: 1 if root.show_back_button else 0
            disabled: not root.show_back_button
"""

class KivyCamera(KivyImage):
    __events__ = ('on_sign_detected',) # Declare the event

    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None
        self.hands_detector = None
        self.fps = 20
        self.is_active = False
        self._initialize_mediapipe()

    def _initialize_mediapipe(self):
        global mp_hands, mp_drawing, mp_drawing_styles
        if mp_hands is None: # Initialize only once
            try:
                import mediapipe as mp
                mp_hands = mp.solutions.hands
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                print("MediaPipe initialized successfully.")
            except ImportError:
                print("Failed to import MediaPipe. Hand tracking will not be available.")
                mp_hands = "Error" # Mark as error
            except Exception as e:
                print(f"Error initializing MediaPipe components: {e}")
                mp_hands = "Error"


        if mp_hands != "Error" and self.hands_detector is None:
            try:
                self.hands_detector = mp_hands.Hands(
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
                print("MediaPipe Hands detector created.")
            except Exception as e:
                print(f"Failed to create MediaPipe Hands detector: {e}")
                self.hands_detector = "Error"


    def start(self):
        if self.hands_detector is None or self.hands_detector == "Error":
            self._initialize_mediapipe() # Try to initialize again if it failed
            if self.hands_detector is None or self.hands_detector == "Error":
                print("MediaPipe Hands detector not available. Cannot start camera for hand tracking.")
                return False

        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                # Try other camera indices
                for i in range(1, 4):
                    self.capture = cv2.VideoCapture(i)
                    if self.capture.isOpened(): break
                if not self.capture.isOpened():
                    print("Cannot open camera.")
                    self.capture = None
                    return False
        self.is_active = True
        Clock.schedule_interval(self.update, 1.0 / self.fps)
        print("KivyCamera started.")
        return True

    def stop(self):
        self.is_active = False
        Clock.unschedule(self.update)
        if self.capture:
            self.capture.release()
            self.capture = None
        # self.texture = None # Avoid clearing texture to prevent black flicker if widget is still visible
        print("KivyCamera stopped.")

    def update(self, dt):
        if not self.is_active or not self.capture or not self.capture.isOpened() or \
           self.hands_detector is None or self.hands_detector == "Error":
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands_detector.process(frame_rgb)
        frame_rgb.flags.writeable = True

        recognized_correctly = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if mp_drawing: # Check if drawing utils are available
                    mp_drawing.draw_landmarks(
                        frame_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            recognized_correctly = True # SIMULATED: Any hand detected is "correct"

        # Mirror effect (horizontal flip)
        mirrored_frame_rgb = cv2.flip(frame_rgb, 1)
        # Convert back to BGR for Kivy texture
        processed_frame_bgr = cv2.cvtColor(mirrored_frame_rgb, cv2.COLOR_RGB2BGR)
        # Vertical flip for Kivy texture orientation
        buf = cv2.flip(processed_frame_bgr, 0).tobytes()

        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = image_texture

        if recognized_correctly:
            self.dispatch('on_sign_detected') # Dispatch the event

    def on_sign_detected(self, *args):
        # This is an event handler that Kivy calls when dispatch('on_sign_detected') is used.
        # It's here so Kivy recognizes the event. The actual logic is in LearningScreen.
        pass

class MainScreen(Screen):
    pass

class VocabMenuScreen(Screen):
    pass

class WordSelectionScreen(Screen):
    pass

class LearningScreen(Screen):
    current_word = StringProperty("")
    video_source = StringProperty("")
    status_message = StringProperty("")

    show_video_player = BooleanProperty(True)
    show_camera_view = BooleanProperty(False)
    show_next_button = BooleanProperty(True)
    show_back_button = BooleanProperty(False)

    video_player = ObjectProperty(None)
    kivy_camera = ObjectProperty(None)
    status_label = ObjectProperty(None)
    next_button = ObjectProperty(None)
    back_to_menu_button = ObjectProperty(None)

    video_paths = {
        "ที่ไหน": "video_where.mp4",
        "สวย": "video_beautiful.mp4",
        "default": "default.mp4" # Fallback video
    }

    def on_enter(self, *args):
        super().on_enter(*args)
        self.status_message = ""
        self.show_video_player = True
        self.show_camera_view = False
        self.show_next_button = True
        self.show_back_button = False

        if self.kivy_camera:
            self.kivy_camera.stop()

        self.video_source = self.video_paths.get(self.current_word, self.video_paths["default"])
        if self.video_player:
            self.video_player.state = 'play'
        print(f"LearningScreen: Entering for word '{self.current_word}', video: {self.video_source}")

    def on_video_end(self, *args):
        print(f"Video '{self.video_source}' ended.")
        self.show_next_button = True # Ensure next button is active

    def prepare_for_camera(self):
        print("LearningScreen: prepare_for_camera called")
        if self.video_player:
            self.video_player.state = 'stop'
        self.show_video_player = False
        self.show_next_button = False

        self.show_camera_view = True
        if self.kivy_camera:
            if not self.kivy_camera.start():
                self.status_message = "ไม่สามารถเปิดกล้องได้"
                self.show_camera_view = False
                self.show_back_button = True # Allow user to go back
                return
            self.status_message = "กรุณาทำท่าทางตามวิดีโอ"
        else:
            self.status_message = "ส่วนแสดงผลกล้องยังไม่พร้อม"
            self.show_back_button = True # Allow user to go back

    def on_sign_recognized_event(self, *args): # Renamed to avoid conflict if Kivy uses on_sign_recognized
        if self.status_message == "ผ่าน!":
            return

        print("LearningScreen: Sign recognized event triggered!")
        self.status_message = "ผ่าน!"
        if self.kivy_camera:
            self.kivy_camera.stop()
        # self.show_camera_view = False # Optionally hide camera
        self.show_back_button = True

    def go_to_word_selection(self, *args):
        if self.kivy_camera:
            self.kivy_camera.stop()
        self.manager.current = 'word_selection'

    def on_leave(self, *args):
        super().on_leave(*args)
        print(f"LearningScreen: Leaving screen for word '{self.current_word}'")
        if self.video_player:
            self.video_player.state = 'stop'
            self.video_source = ""
        if self.kivy_camera:
            self.kivy_camera.stop()
        self.status_message = ""


class SignLanguageApp(App):
    def build(self):
        Builder.load_string(KV_STRING)
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(VocabMenuScreen(name='vocab_menu'))
        sm.add_widget(WordSelectionScreen(name='word_selection'))
        sm.add_widget(LearningScreen(name='learning'))
        return sm

    def select_word(self, word):
        learning_screen = self.root.get_screen('learning')
        learning_screen.current_word = word
        self.root.current = 'learning'

def create_dummy_files():
    # Create a simple placeholder logo.png if it doesn't exist
    if not os.path.exists("logo.png"):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (200, 100), color = (73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((50,30), "LOGO", fill=(255,255,0), font_size=40)
            img.save('logo.png')
            print("Created dummy logo.png")
        except ImportError:
            print("Pillow library not found. Cannot create dummy logo.png. Please create 'logo.png' manually.")
        except Exception as e:
            print(f"Error creating dummy logo.png: {e}")

    # Create dummy video files if they don't exist
    dummy_video_files = ["video_where.mp4", "video_beautiful.mp4", "default.mp4"]
    for vid_file in dummy_video_files:
        if not os.path.exists(vid_file):
            try:
                with open(vid_file, 'w') as f:
                    f.write("") # Create an empty file
                print(f"Created dummy video file: {vid_file} (This is an empty file, replace with actual video)")
            except Exception as e:
                print(f"Error creating dummy video file {vid_file}: {e}")


if __name__ == '__main__':
    create_dummy_files()
    SignLanguageApp().run()
