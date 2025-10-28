#	pip3 install opencv-python
#	pip3 install ultralytics
#	pip3 install EmailMessage
#	pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# lastest microsoft runtime library
# https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version

import cv2
from ultralytics import YOLO
import datetime
import datetime
import smtplib
from email.message import EmailMessage
import os
import sys
import time

email_me = True
email_conf = 0.50
image_save_conf = 0.50

# Local imports
my_modules_path = os.getcwd()
if sys.path[0] != my_modules_path:
    sys.path.insert(0, my_modules_path)

# The email_configuration file contains email information that will not be included in github files
# === CONFIGURATION ===
#EMAIL_SENDER   = "s@gmail.com"
#EMAIL_PASSWORD = "x"   # Use an "App Password" if Gmail
#EMAIL_RECEIVER = "r@gmail.com"

from email_configuration import *

# Function to send email with image
def send_email(image_path, label):
    msg = EmailMessage()
    msg["Subject"] = f"Detection Alert: {label}"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content(f"A {label} was detected. See attached image.")

    # Attach image
    with open(image_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(image_path)
    msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)

    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)
    print(f"📧 Email sent with {image_path}")


detection_items = ["person"] #, "car", "truck", "motorcycle"]):
email_items	    = ["person"]

print(f"Detected {email_items} will be E-mailed to {EMAIL_SENDER} from {EMAIL_RECEIVER}")
print("loading model")
# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")  # 'n' = nano (fastest), can use yolov8s/m/l for better accuracy

print("open webcam, this will take a moment...")
# Open webcam
cap = cv2.VideoCapture(0)

last_email_time = 0.0
time_between_emails = 60


    
print("Starting Detection")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])  # class ID
            conf = float(box.conf[0])  # confidence
            label = model.names[cls_id]  # class label
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if conf > image_save_conf and conf < email_conf and label in detection_items:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"{timestamp}: {label} Confidence: {conf}")
                
            # Only care about people and vehicles
            if (conf >= email_conf) and (label in detection_items):
                # Get bounding box
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f} {timestamp}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save frame wqhen detected
                filename = f"Captures\\{timestamp}_{label}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Captured: {filename}")
                
                if label in email_items:
                    current_time = time.time()
                    elapsed_time = current_time - last_email_time
                        
                    if elapsed_time >= time_between_emails:
                        last_email_time = time.time()
                        try:
                            if email_me:
                                send_email(filename, label)
                        except Exception as e:
                            print(f"{timestamp}: {str(e)}")
                    else:
                        print("Not sending email")

    cv2.imshow("Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
