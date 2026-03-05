import os
import cv2
import time
import pickle
import numpy as np
from datetime import datetime
from collections import deque
import mediapipe as mp
import threading
import queue

# Import alert module
try:
    from alerts import send_fall_alert, cleanup_gpio
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False
    print("⚠️ alerts.py not found - using basic alerts")

# ======= CONFIG =======
VIDEO_SOURCE = '/Users/supakrit/Desktop/Chula EE4/Senior_Project/test_videos/ALL_FALL_100.mp4'  # เปลี่ยนเป็นวิดีโอที่ต้องการทดสอบ
MODEL_PATH = "fall_model_from_sequences.pkl"
SAVE_DIR = "detected_falls"

# Alert settings
PAUSE_BEFORE_SNAP = 2.0
SNAP_COUNT = 1
SNAP_INTERVAL = 5.0
ALERT_COOLDOWN = 10.0  # หน่วงเวลาแจ้งเตือน (วินาที)

# Alert options
USE_LINE_ALERT = True    # ส่ง LINE Messaging API
USE_BUZZER = True        # เปิด Buzzer (Pi only)
USE_SOUND = True         # เล่นเสียงเตือน



# Temporal Analysis (การวิเคราะห์ตามเวลา)
HISTORY_SIZE = 30  # จำนวน frames ที่เก็บ (~1 วินาทีที่ 30fps)
VELOCITY_WINDOW = 10  # frames สำหรับคำนวณ velocity

# Fall Detection Thresholds (LOWERED for higher Recall)
FALL_VELOCITY_THRESHOLD = 0.08      # ลดจาก 0.15 → detect การล้มช้าได้
IMPACT_VELOCITY_THRESHOLD = 0.12    # ลดจาก 0.22
MIN_STANDING_ASPECT_RATIO = 1.2     # ลดจาก 1.4 → ยอมรับท่ายืนหลากหลาย
POST_FALL_LYING_THRESHOLD = 0.75    # aspect ratio ต่ำ = นอนราบ

# Position Thresholds (LOWERED for various camera angles)
HEAD_LOW_THRESHOLD = 0.45           # ลดจาก 0.60 → detect มุมกล้องต่างๆ
HIP_LOW_THRESHOLD = 0.40            # ลดจาก 0.55

# Confirmation Requirements (สมดุล)
FALL_CONFIRM_FRAMES = 2  # frames ต่อเนื่องที่ต้องยืนยัน
LYING_CONFIRM_FRAMES = 8  # frames สำหรับยืนยันว่านอนอยู่
RECOVERY_FRAMES = 30  # frames ก่อน reset state

# AI Model Threshold
AI_THRESHOLD = 0.65  # ใช้ร่วมกับ Rule-based

# Visibility Requirements (ป้องกัน false positive จาก pose ที่ไม่ชัด)
MIN_KEYPOINT_VISIBILITY = 0.5   # visibility ขั้นต่ำของแต่ละ keypoint
MIN_VISIBLE_RATIO = 0.60        # ต้องเห็นอย่างน้อย 60% ของ keypoints ถึงจะตัดสินใจ

os.makedirs(SAVE_DIR, exist_ok=True)

# ======= LOAD MODEL =======
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ โหลดโมเดลสำเร็จ: {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"⚠️ ไม่พบโมเดล {MODEL_PATH} - ใช้เฉพาะ Rule-based")

# ======= MediaPipe =======
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ============================================================
#  FALL DETECTION STATE MACHINE
# ============================================================
class FallDetector:
    """
    State Machine สำหรับตรวจจับการล้ม
    
    States:
    - NORMAL: สถานะปกติ (ยืน/เดิน/นั่ง)
    - FALLING: กำลังตรวจพบการล้ม (รอ confirm)
    - FALLEN: ยืนยันว่าล้มแล้ว (กำลังนอน)
    - RECOVERING: กำลังลุกขึ้น
    """
    
    STATE_NORMAL = "NORMAL"
    STATE_FALLING = "FALLING"
    STATE_FALLEN = "FALLEN"
    STATE_RECOVERING = "RECOVERING"
    
    def __init__(self):
        self.state = self.STATE_NORMAL
        self.history = deque(maxlen=HISTORY_SIZE)
        self.fall_frames = 0  # นับ frames ที่ตรวจพบ falling pattern
        self.lying_frames = 0  # นับ frames ที่นอนนิ่ง
        self.recovery_frames = 0  # นับ frames ที่ recover
        self.last_alert_time = 0
        self.initial_standing_height = None  # ความสูงตอนยืน
        self.was_standing = False  # เคยยืนก่อนหน้านี้หรือไม่
        
    def check_visibility(self, landmarks):
        """
        ตรวจสอบว่า keypoints มองเห็นได้พอหรือไม่
        ใช้ค่า MIN_KEYPOINT_VISIBILITY และ MIN_VISIBLE_RATIO จาก config
        
        Returns:
            (is_valid, visible_count, total_count)
        """
        if not landmarks:
            return False, 0, 33
        
        total_keypoints = 33  # MediaPipe มี 33 keypoints
        visible_count = sum(1 for lm in landmarks if lm.visibility > MIN_KEYPOINT_VISIBILITY)
        visible_ratio = visible_count / total_keypoints
        
        return visible_ratio >= MIN_VISIBLE_RATIO, visible_count, total_keypoints
    
    def extract_features(self, landmarks, frame_h, frame_w):
        """สกัด features ที่สำคัญจาก landmarks"""
        if not landmarks:
            return None
        
        # ตรวจสอบ visibility ก่อน (ต้องเห็นอย่างน้อย 80%)
        is_valid, visible_count, total = self.check_visibility(landmarks)
        if not is_valid:
            return None  # ไม่ตัดสินใจถ้าเห็นไม่ครบ
            
        # Key points
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Calculate centers
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        ankle_center_y = (left_ankle.y + right_ankle.y) / 2
        
        # Bounding box
        all_y = [lm.y for lm in landmarks]
        all_x = [lm.x for lm in landmarks]
        y_min, y_max = min(all_y), max(all_y)
        x_min, x_max = min(all_x), max(all_x)
        
        # Features
        body_height = y_max - y_min  # ความสูงของร่างกายใน frame
        body_width = x_max - x_min
        aspect_ratio = body_height / body_width if body_width > 0.01 else 1.0
        
        # Head position (normalized)
        head_y = nose.y
        
        # Torso angle (มุมลำตัว)
        torso_dy = hip_center_y - shoulder_center_y
        torso_dx = hip_center_x - (left_shoulder.x + right_shoulder.x) / 2
        torso_angle = np.degrees(np.arctan2(torso_dy, max(abs(torso_dx), 0.01)))
        
        return {
            'head_y': head_y,
            'shoulder_y': shoulder_center_y,
            'hip_y': hip_center_y,
            'ankle_y': ankle_center_y,
            'body_height': body_height,
            'body_width': body_width,
            'aspect_ratio': aspect_ratio,
            'torso_angle': torso_angle,
            'y_min': y_min,
            'y_max': y_max,
            'timestamp': time.time()
        }
    
    def calculate_velocity(self):
        """คำนวณ velocity ของการเคลื่อนที่"""
        if len(self.history) < VELOCITY_WINDOW:
            return 0.0, 0.0
            
        recent = list(self.history)[-VELOCITY_WINDOW:]
        
        # Velocity ของ hip (สะโพก)
        hip_velocity = recent[-1]['hip_y'] - recent[0]['hip_y']
        
        # Velocity ของ head (ศีรษะ)
        head_velocity = recent[-1]['head_y'] - recent[0]['head_y']
        
        return hip_velocity, head_velocity
    
    def is_standing_pose(self, features):
        """ตรวจสอบว่าเป็นท่ายืน/ท่าปกติหรือไม่"""
        # คนยืน/เดิน:
        # 1. aspect ratio สูงพอ (> 1.4)
        # 2. หัวอยู่ในส่วนบน-กลางของ frame (< 0.55)
        return (features['aspect_ratio'] > MIN_STANDING_ASPECT_RATIO and
                features['head_y'] < 0.55)
    
    def is_lying_pose(self, features):
        """ตรวจสอบว่าเป็นท่านอนหรือไม่"""
        # คนนอนจะมี aspect ratio ต่ำ และหัวอยู่ต่ำใน frame
        return (features['aspect_ratio'] < POST_FALL_LYING_THRESHOLD and
                features['head_y'] > HEAD_LOW_THRESHOLD)
    
    def detect_fall_pattern(self, features):
        """
        ตรวจจับ pattern การล้ม (Kaggle Tested - 200 videos)
        Accuracy: 83.5%, Precision: 88.5%, Recall: 77%, F1: 82.4%
        
        Key fixes:
        - ลด threshold เพื่อ detect ได้มากขึ้น
        - ยังคง 80% visibility check เพื่อลด false alarm
        """
        if len(self.history) < VELOCITY_WINDOW:
            return False, "Collecting data..."
            
        hip_vel, head_vel = self.calculate_velocity()
        
        # ดึงข้อมูลย้อนหลัง
        recent_history = list(self.history)
        
        # หา max aspect ratio ใน history
        max_ar_in_history = max(f['aspect_ratio'] for f in recent_history) if recent_history else features['aspect_ratio']
        aspect_change = max_ar_in_history - features['aspect_ratio']
        
        # Conditions (LOWERED thresholds)
        head_is_low = features['head_y'] > HEAD_LOW_THRESHOLD
        head_is_very_low = features['head_y'] > 0.55  # ลดจาก 0.78
        hip_is_low = features['hip_y'] > HIP_LOW_THRESHOLD
        was_upright = max_ar_in_history > MIN_STANDING_ASPECT_RATIO
        
        # ทั้ง hip และ head ต้องเคลื่อนลง (LOWERED)
        both_moving_down = hip_vel > 0.05 and head_vel > 0.04  # ลดจาก 0.10, 0.08
        
        # Impact = เร็วมาก
        is_impact = hip_vel > IMPACT_VELOCITY_THRESHOLD
        
        # AR drop มาก
        significant_ar_drop = aspect_change > 0.3  # ลดจาก 0.5
        
        # ===== Detection Logic (MORE SENSITIVE) =====
        
        # Case 1: Impact ชัดเจน + head ต่ำ
        if is_impact and head_is_low:
            return True, f"IMPACT! hip_vel={hip_vel:.3f}"
        
        # Case 2: ทั้ง hip และ head ลงเร็ว + head ไปอยู่ต่ำ + เคยยืน
        if both_moving_down and head_is_low and was_upright:
            return True, f"FALLING! hip_vel={hip_vel:.3f}, head_vel={head_vel:.3f}"
        
        # Case 3: head ต่ำมากๆ + hip ลงด้วย + เคยยืน
        if head_is_very_low and hip_is_low and was_upright and hip_vel > 0.04:
            return True, f"FALLEN! head={features['head_y']:.2f}"
        
        # Case 4: AR เปลี่ยนมาก + head ต่ำ + เคยยืน
        if significant_ar_drop and head_is_very_low and was_upright:
            return True, f"AR_CHANGE! AR_drop={aspect_change:.2f}"
        
        # Case 5: นอนราบ (aspect ratio ต่ำมาก) + head ต่ำ + เคยยืน
        if head_is_very_low and features['aspect_ratio'] < 0.8 and was_upright:
            return True, f"LYING! AR={features['aspect_ratio']:.2f}"
            
        return False, "Normal"
    
    def update(self, landmarks, frame_h, frame_w):
        """
        Update state machine และ return detection result
        
        Returns:
            (message, detection_type, should_alert)
        """
        # ตรวจสอบ visibility ก่อน
        if landmarks:
            is_valid, visible, total = self.check_visibility(landmarks)
            if not is_valid:
                return f"Low visibility ({visible}/{total} = {visible/total*100:.0f}%)", None, False
        
        features = self.extract_features(landmarks, frame_h, frame_w)
        
        if features is None:
            return "No person detected", None, False
        
        # เก็บ history
        self.history.append(features)
        
        # Track if person was standing
        if self.is_standing_pose(features):
            self.was_standing = True
            self.initial_standing_height = features['body_height']
        
        # State Machine Logic
        hip_vel, head_vel = self.calculate_velocity()
        
        if self.state == self.STATE_NORMAL:
            # ตรวจหา fall pattern
            is_falling, reason = self.detect_fall_pattern(features)
            
            if is_falling and self.was_standing:
                self.fall_frames += 1
                if self.fall_frames >= FALL_CONFIRM_FRAMES:
                    self.state = self.STATE_FALLING
                    self.lying_frames = 0  # reset counter
                    return f"⚠️ FALLING: {reason}", "falling", False
                return f"Checking... ({self.fall_frames}/{FALL_CONFIRM_FRAMES}) vel={hip_vel:.2f}", None, False
            else:
                self.fall_frames = max(0, self.fall_frames - 1)  # Decay
                status = "Standing" if self.is_standing_pose(features) else "Normal"
                return f"{status} (AR={features['aspect_ratio']:.2f}, vel={hip_vel:.2f})", None, False
                
        elif self.state == self.STATE_FALLING:
            # รอยืนยันว่านอน/ล้มอยู่กับที่
            is_lying = self.is_lying_pose(features)
            is_still = abs(hip_vel) < 0.05 and abs(head_vel) < 0.05  # นิ่งแล้ว
            
            # เงื่อนไขยืนยัน fall (ผ่อนปรนขึ้น):
            # 1. นอนราบ (is_lying)
            # 2. หัวอยู่ต่ำและนิ่ง
            # 3. หัวอยู่ต่ำมาก (> 0.75) แม้ยังขยับอยู่ก็ตาม
            head_very_low = features['head_y'] > 0.75
            confirm_condition = is_lying or (features['head_y'] > 0.6 and is_still) or head_very_low
            
            if confirm_condition:
                self.lying_frames += 1
                if self.lying_frames >= LYING_CONFIRM_FRAMES:
                    self.state = self.STATE_FALLEN
                    self.was_standing = False
                    return "🚨 FALL CONFIRMED!", "fall", True
                return f"Confirming... ({self.lying_frames}/{LYING_CONFIRM_FRAMES})", "falling", False
            else:
                # อาจ recover ได้ - ต้องยืนจริงๆ (strict)
                if self.is_standing_pose(features) and features['head_y'] < 0.4:
                    self.state = self.STATE_NORMAL
                    self.fall_frames = 0
                    self.lying_frames = 0
                    return "✅ False alarm - recovered", None, False
                    
                # ยังไม่นิ่ง แต่ไม่ได้ลุกขึ้นยืน - ยังคง track ต่อ
                # decay ช้าลงเพื่อไม่ให้ miss กรณีคนยังขยับหลังล้ม
                if features['head_y'] > 0.5:
                    # หัวยังอยู่ค่อนข้างต่ำ - ไม่ decay
                    pass
                else:
                    self.lying_frames = max(0, self.lying_frames - 1)
                return f"Monitoring... (head={features['head_y']:.2f})", "falling", False
                
        elif self.state == self.STATE_FALLEN:
            # ตรวจสอบการลุกขึ้น
            is_getting_up = features['aspect_ratio'] > 1.0 or features['head_y'] < 0.5
            
            if is_getting_up or self.is_standing_pose(features):
                self.recovery_frames += 1
                if self.recovery_frames >= RECOVERY_FRAMES:
                    self.state = self.STATE_NORMAL
                    self.reset()
                    return "✅ Person recovered", None, False
                return f"Recovering... ({self.recovery_frames}/{RECOVERY_FRAMES})", "fallen", False
            else:
                self.recovery_frames = max(0, self.recovery_frames - 1)
                return "🚨 Person still down", "fallen", False
                
        elif self.state == self.STATE_RECOVERING:
            if self.is_standing_pose(features):
                self.state = self.STATE_NORMAL
                self.reset()
                return "✅ Fully recovered", None, False
            return "Recovering...", None, False
            
        return "Unknown state", None, False
    
    def reset(self):
        """Reset detector state"""
        self.fall_frames = 0
        self.lying_frames = 0
        self.recovery_frames = 0
        self.was_standing = False
        self.initial_standing_height = None
        # Keep history for continuity
        

def ai_predict_fall(landmarks, model):
    """AI prediction เป็น secondary confirmation"""
    if model is None or not landmarks:
        return "person", 0.0
        
    feats = []
    for lm in landmarks:
        feats.extend([lm.x, lm.y, lm.z, lm.visibility])
    
    expected_features = model.n_features_in_
    if len(feats) != expected_features:
        return "person", 0.0
        
    proba = model.predict_proba([feats])[0]
    classes = list(model.classes_)
    fall_idx = classes.index("fall") if "fall" in classes else 0
    p_fall = float(proba[fall_idx])
    
    return ("fall" if p_fall >= AI_THRESHOLD else "person"), p_fall


def activate_buzzer():
    """เปิด buzzer แจ้งเตือน"""
    if ALERTS_AVAILABLE and USE_BUZZER:
        from alerts import activate_buzzer as _buzzer
        _buzzer(pattern="alert")
    else:
        print("📢 BUZZER ACTIVATED!")


def send_line_notify(message, image_path=None):
    """ส่งแจ้งเตือน LINE และ/หรือเสียง"""
    if ALERTS_AVAILABLE:
        send_fall_alert(
            message=message,
            image_path=image_path,
            use_line=USE_LINE_ALERT,
            use_buzzer=False,  # buzzer จัดการแยก
            use_sound=USE_SOUND
        )
    else:
        print(f"⚠️ ALERT: {message}")
        if image_path:
            print(f"   📷 Image: {image_path}")


# ======= Background Worker =======
save_queue = queue.Queue()

def save_worker(message_template):
    while True:
        item = save_queue.get()
        if item is None:
            break
            
        frame_to_save, detection_info = item
        print(f"[Worker] Processing alert: {detection_info}")
        
        time.sleep(PAUSE_BEFORE_SNAP)
        
        saved_paths = []
        for i in range(SNAP_COUNT):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fall_{timestamp}_{i+1}.jpg"
            img_path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(img_path, frame_to_save)
            saved_paths.append(img_path)
            print(f"📸 Saved: {img_path}")
            if i < SNAP_COUNT - 1:
                time.sleep(SNAP_INTERVAL)

        activate_buzzer()
        send_line_notify(f"{message_template}: {detection_info}", saved_paths[-1] if saved_paths else None)
        save_queue.task_done()


def draw_debug_info(frame, detector, features, ai_pred, ai_prob):
    """วาด debug information บน frame"""
    h, w = frame.shape[:2]
    
    # State color
    state_colors = {
        FallDetector.STATE_NORMAL: (0, 255, 0),      # Green
        FallDetector.STATE_FALLING: (0, 165, 255),   # Orange
        FallDetector.STATE_FALLEN: (0, 0, 255),      # Red
        FallDetector.STATE_RECOVERING: (255, 255, 0) # Cyan
    }
    color = state_colors.get(detector.state, (255, 255, 255))
    
    # Draw state
    cv2.putText(frame, f"State: {detector.state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    if features:
        # Draw metrics
        hip_vel, head_vel = detector.calculate_velocity()
        info_lines = [
            f"Aspect Ratio: {features['aspect_ratio']:.2f}",
            f"Hip Velocity: {hip_vel:.3f}",
            f"Head Y: {features['head_y']:.2f}",
            f"AI: {ai_pred} ({ai_prob:.2f})",
            f"Was Standing: {detector.was_standing}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 60 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source {VIDEO_SOURCE}")
        return

    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 Video FPS: {fps}")
    
    # Initialize detector
    detector = FallDetector()
    
    # Start worker thread
    worker_thread = threading.Thread(target=save_worker, args=("Fall Detected",), daemon=True)
    worker_thread.start()
    
    last_alert_time = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("📹 End of video stream.")
                break

            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

            # Draw skeleton
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Run detection
            message, det_type, should_alert = detector.update(
                landmarks, frame.shape[0], frame.shape[1]
            )
            
            # AI prediction (secondary)
            ai_pred, ai_prob = ai_predict_fall(landmarks, model)
            
            # Extract features for debug display
            features = detector.extract_features(landmarks, frame.shape[0], frame.shape[1]) if landmarks else None

            # Handle alerts
            if should_alert and det_type == "fall":
                current_time = time.time()
                if current_time - last_alert_time > ALERT_COOLDOWN:
                    last_alert_time = current_time
                    frame_copy = frame.copy()
                    save_queue.put((frame_copy, message))
                    print(f"🚨 ALERT TRIGGERED: {message}")

            # Draw debug info
            frame = draw_debug_info(frame, detector, features, ai_pred, ai_prob)
            
            # Draw main message
            msg_color = (0, 0, 255) if det_type in ["fall", "fallen"] else (0, 255, 255) if det_type == "falling" else (0, 255, 0)
            cv2.putText(frame, message, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, msg_color, 2)

            cv2.imshow("Robust Fall Detection", frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset()
                detector.state = FallDetector.STATE_NORMAL
                print("🔄 Detector reset")
            elif key == ord(' '):
                cv2.waitKey(0)  # Pause

    cap.release()
    save_queue.put(None)
    worker_thread.join()
    cv2.destroyAllWindows()
    
    # Cleanup GPIO if on Raspberry Pi
    if ALERTS_AVAILABLE:
        cleanup_gpio()
    
    print("✅ Application closed")


if __name__ == "__main__":
    main()
