"""
Alert Module for Fall Detection System
- LINE Messaging API (LINE Notify ปิดบริการแล้ว)
- Buzzer (GPIO for Raspberry Pi)
- Local sound alert (Mac/PC)
- Imgur image upload (สำหรับส่งรูปผ่าน LINE)
"""

import os
import requests
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# LINE Messaging API
LINE_CHANNEL_ACCESS_TOKEN = "EcoHFbJZdIXA1WsodjnqpArbhzhmzG4/yPjpZkcsYc2YzdbMmwVU/ZJYva72uJExUBGMZ2V2wiyiSR0943XIirF5cxo5ehCPjyl+vIrgq6pg+4oHBUZMV7D4yngTy4LvlrmSp2L1s7KFRI6uHENm+QdB04t89/1O/w1cDnyilFU="
LINE_USER_ID = "U4d687f09d53f6ba70abc2da505c26907"


# Buzzer GPIO Pin (สำหรับ Raspberry Pi)
BUZZER_PIN = 27

# ============================================================
# IMAGE UPLOAD - Google Drive via rclone
# ============================================================

# Google Drive Folder ID (จาก URL: https://drive.google.com/drive/folders/XXXXX)
GOOGLE_DRIVE_FOLDER_ID = "1JXUcbd2WiKkzI8jedZrM_JjThepp4fsU"
RCLONE_REMOTE = "gdrive"  # ชื่อ remote ที่ตั้งใน rclone config

def upload_image(image_path):
    """
    Upload รูปไป Google Drive ด้วย rclone
    
    Setup บน Pi:
    1. sudo apt install rclone
    2. rclone config → สร้าง remote ชื่อ "gdrive" เชื่อม Google Drive
    3. ทดสอบ: rclone lsd gdrive:
    
    Args:
        image_path: path ไปยังไฟล์รูป
    
    Returns:
        str: URL ของรูป หรือ None ถ้าล้มเหลว
    """
    import subprocess
    
    if not image_path or not os.path.exists(image_path):
        return None
    
    try:
        file_name = os.path.basename(image_path)
        
        # Upload ด้วย rclone
        # rclone copy local_file gdrive:folder_id/
        cmd = [
            "rclone", "copy",
            image_path,
            f"{RCLONE_REMOTE}:detected_falls",
            "--drive-root-folder-id", GOOGLE_DRIVE_FOLDER_ID
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"✅ Image uploaded to Google Drive: {file_name}")
            # ไม่มี direct URL แต่ดูได้ใน Drive folder
            return f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}"
        else:
            print(f"❌ rclone error: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("❌ rclone not installed. Run: sudo apt install rclone")
        return None
    except subprocess.TimeoutExpired:
        print("❌ Upload timeout")
        return None
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None


# ============================================================
# LINE MESSAGING API
# ============================================================

def send_line_message(message, image_url=None):
    """
    ส่งข้อความผ่าน LINE Messaging API
    (รูปเก็บไว้ใน Google Drive แทน)
    """
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        print("⚠️ LINE Messaging API not configured")
        print(f"   Message: {message}")
        return False
    
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {LINE_CHANNEL_ACCESS_TOKEN}'
    }
    
    # สร้าง timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"🚨 Fall Detection Alert\n⏰ {timestamp}\n\n{message}\n\n📷 ดูรูปได้ที่ Google Drive"
    
    data = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": full_message}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ LINE message sent: {message[:50]}...")
            return True
        else:
            print(f"❌ LINE API error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ LINE API error: {e}")
        return False


def send_line_image_file(message, image_path):
    """
    ส่งข้อความ (รูปเก็บไว้ใน Google Drive แทน)
    """
    # Upload รูปไป Google Drive (ดูได้ทีหลัง)
    if image_path and os.path.exists(image_path):
        print(f"📤 Uploading image to Drive: {image_path}")
        upload_image(image_path)
    
    # ส่งแค่ข้อความ (ไม่ส่งรูปผ่าน LINE)
    result = send_line_message(message, image_url=None)
    
    return result


# ============================================================
# BUZZER (Raspberry Pi GPIO)
# ============================================================

IS_RASPBERRY_PI = False
GPIO = None

try:
    import RPi.GPIO as GPIO
    IS_RASPBERRY_PI = True
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    print(f"✅ GPIO initialized (Buzzer on pin {BUZZER_PIN})")
except ImportError:
    print("ℹ️ RPi.GPIO not available (not on Raspberry Pi)")
except Exception as e:
    print(f"⚠️ GPIO setup error: {e}")


def activate_buzzer(duration=2.0, pattern="alert"):
    """
    เปิด buzzer แจ้งเตือน
    
    Args:
        duration: ระยะเวลา (วินาที)
        pattern: "alert" (beep 3 ครั้ง), "continuous"
    """
    if IS_RASPBERRY_PI and GPIO:
        try:
            import time
            if pattern == "alert":
                for _ in range(3):
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    time.sleep(0.3)
                    GPIO.output(BUZZER_PIN, GPIO.LOW)
                    time.sleep(0.2)
            else:
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                time.sleep(duration)
                GPIO.output(BUZZER_PIN, GPIO.LOW)
            
            print(f"🔔 Buzzer activated ({pattern})")
            return True
        except Exception as e:
            print(f"❌ Buzzer error: {e}")
            return False
    else:
        print(f"🔔 [BUZZER SIM] Alert! ({pattern})")
        return True


def cleanup_gpio():
    """Cleanup GPIO on exit"""
    if IS_RASPBERRY_PI and GPIO:
        GPIO.cleanup()
        print("✅ GPIO cleaned up")


# ============================================================
# LOCAL SOUND (Mac/PC)
# ============================================================

def play_alert_sound():
    """เล่นเสียงแจ้งเตือนบน Mac/PC"""
    import platform
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            os.system('afplay /System/Library/Sounds/Sosumi.aiff &')
            print("🔊 Mac alert sound played")
        elif system == "Linux":
            os.system('aplay /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null &')
            print("🔊 Linux alert sound played")
        elif system == "Windows":
            import winsound
            winsound.Beep(1000, 500)
            print("🔊 Windows alert sound played")
        return True
    except Exception as e:
        print(f"⚠️ Sound error: {e}")
        return False


# ============================================================
# COMBINED ALERT FUNCTION
# ============================================================

def send_fall_alert(message, image_path=None, use_line=True, use_buzzer=True, use_sound=True):
    """
    ส่งแจ้งเตือนทุกช่องทาง
    """
    results = {}
    
    print("\n" + "="*50)
    print("🚨 FALL ALERT TRIGGERED!")
    print("="*50)
    
    if use_buzzer:
        results['buzzer'] = activate_buzzer(pattern="alert")
    
    if use_sound:
        results['sound'] = play_alert_sound()
    
    if use_line:
        results['line'] = send_line_image_file(message, image_path)
    
    print("="*50 + "\n")
    
    return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("="*50)
    print("Testing Alert Module")
    print("="*50)
    
    # Test LINE
    print("\n[1] Testing LINE Messaging API...")
    send_line_message("Test: Fall Detection System is working!")
    
    # Test Buzzer
    print("\n[2] Testing Buzzer...")
    activate_buzzer(pattern="alert")
    
    # Test Sound
    print("\n[3] Testing Sound...")
    play_alert_sound()
    
    # Test Combined
    print("\n[4] Testing Combined Alert...")
    send_fall_alert("Test: Person fell down!")
    
    cleanup_gpio()
    
    print("\n" + "="*50)
    print("Setup Guide:")
    print("="*50)
    print("""
=== LINE Messaging API ===
1. ไปที่ https://developers.line.biz/console/
2. สร้าง Provider → Channel (Messaging API)
3. Copy "Channel access token (long-lived)"
4. Scan QR เพิ่ม Bot เป็นเพื่อน
5. หา User ID จาก LINE Official Account Manager

=== Environment Variables ===
export LINE_CHANNEL_ACCESS_TOKEN="your_line_token"
export LINE_USER_ID="your_user_id"

=== Google Drive via rclone (บน Pi) ===
1. sudo apt install rclone
2. rclone config → สร้าง remote "gdrive"
3. รูปจะ upload ไป folder ที่ตั้งไว้อัตโนมัติ
""")
