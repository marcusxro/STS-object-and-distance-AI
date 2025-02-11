import cv2
import pyttsx3
import threading
import numpy as np
from ultralytics import YOLO

path_model = YOLO("yolov8s-seg.pt") 
obj_model = YOLO("yolov8n.pt")  
class_names = obj_model.names  

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait(), daemon=True).start()

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)

FOCAL_LENGTH = 500 
frame_skip = 3  
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    path_results = path_model(img_rgb, conf=0.25)
    obj_results = obj_model(img, stream=True)

    mask = np.zeros_like(img)
    left_objects, right_objects, front_objects = [], [], []
    grabbable_objects = []

    for r in path_results:
        if r.masks is not None:
            for seg in r.masks.xy:
                pts = np.array([seg], np.int32)
                cv2.polylines(mask, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = img.shape[1] // 2, img.shape[0] - 50
                cv2.line(mask, (img.shape[1] // 2, img.shape[0]), (cx, cy), (0, 0, 255), 3)

    for r in obj_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width_px = x2 - x1
            x_center = (x1 + x2) // 2
            class_name = class_names.get(int(box.cls[0]), "unknown")
            distance_cm = (50 * FOCAL_LENGTH) / width_px
            distance_m = distance_cm / 100

            if x_center < 100:
                left_objects.append((class_name, distance_m))
            elif x_center > 220:
                right_objects.append((class_name, distance_m))
            else:
                front_objects.append((class_name, distance_m))
                if distance_cm <= 50 and width_px > 30:
                    grabbable_objects.append(class_name)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, f"Grabbable: {class_name}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{class_name} {distance_m:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    min_safe_distance = 1.0
    safe_front = all(dist > min_safe_distance for _, dist in front_objects)
    object_warnings = []
    if front_objects:
        object_warnings.append(f"Obstacle ahead: {', '.join(f'{name} at {dist:.1f}m' for name, dist in front_objects)}")
    if left_objects:
        object_warnings.append(f"Obstacle to the left: {', '.join(f'{name} at {dist:.1f}m' for name, dist in left_objects)}")
    if right_objects:
        object_warnings.append(f"Obstacle to the right: {', '.join(f'{name} at {dist:.1f}m' for name, dist in right_objects)}")
    
    navigation_instruction = "Move forward." if safe_front else "Obstacle ahead. Find an opening."
    speak(navigation_instruction)
    if object_warnings:
        speak(". ".join(object_warnings))
    if grabbable_objects:
        speak(f"You can grab a {', '.join(grabbable_objects)} in front of you.")

    combined = cv2.addWeighted(img, 1, mask, 0.5, 0)
    cv2.imshow("Navigation", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
