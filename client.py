import cv2
import subprocess
import requests
import time
from datetime import datetime
import os

# Load OpenCV Haar cascade face detector
face_cascade = cv2.CascadeClassifier("/home/ourRasp/haarcascade_frontalface_default.xml")

# Flask server URL
SERVER_URL = "http://192.168.68.83:5000/upload"

# Image capture settings
IMAGE_WIDTH = "640"
IMAGE_HEIGHT = "480"
IMAGE_PATH = "/home/ourRasp/last_capture.jpg"

def capture_image():
    """Capture image using libcamera-jpeg"""
    try:
        print("Capturing image...")
        result = subprocess.run(
            ["libcamera-jpeg", "-o", IMAGE_PATH, "--width", IMAGE_WIDTH, "--height", IMAGE_HEIGHT],
            check=True,
            capture_output=True,
            text=True
        )
        print("Capture successful")
    except subprocess.CalledProcessError as e:
        print("Failed to capture image")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)


def detect_and_send_faces():
    """Detect faces and send to server"""
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("Failed to read captured image")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected.")
        return

    print(f"Detected {len(faces)} face(s). Sending to server...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (x, y, w, h) in enumerate(faces):
        face_img = image[y:y+h, x:x+w]

        # ? Resize face image to improve recognition by Dlib
        try:
            face_img = cv2.resize(face_img, (150, 150))
        except Exception as e:
            print(f"Error resizing face {i+1}: {e}")
            continue

        #face_path = f"/home/ourRasp/face_{timestamp}_{i}.jpg"
        face_path = f"/home/ourRasp/face_{timestamp}.jpg"
        cv2.imwrite(face_path, face_img)

        # Send to server
        try:
            with open(face_path, 'rb') as img_file:
                files = {'image': img_file}
                response = requests.post(SERVER_URL, files=files)
                print(f"? Sent face {i+1}: {response.text}")
        except Exception as e:
            print("Error sending image:", e)

        # Optional: clean up saved face image
        os.remove(face_path)

def main():
    print("Starting face detection loop using libcamera...")
    try:
        while True:
            capture_image()
            detect_and_send_faces()
            time.sleep(3)  # Capture every 3 seconds
    except KeyboardInterrupt:
        print("Stopping...")

if _name_ == "_main_":
    main()