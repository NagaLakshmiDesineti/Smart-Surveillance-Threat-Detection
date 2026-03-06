import cv2
import cvzone
from ultralytics import YOLO
import smtplib

# Load YOLOv8 model
model = YOLO('yolov8n.pt') 

def send_alert():
    # Email alert logic ikkada untundi
    print("Alert: Threat Detected! Sending Email...")

cap = cv2.VideoCapture(0) # Camera open chestundi

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            
            # If 'person' or 'knife' (threats) detected
            if model.names[cls] == 'person' and conf > 0.5:
                send_alert()
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), 
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0,0,255), 2)

    cv2.imshow("Surveillance", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
