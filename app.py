import cv2
import cvzone
import math
import time
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "pistol", "rifle", "gun"]

# Object sizes (known real-world widths in cm)
object_sizes = {
    "cell phone": 7,
    "bottle": 7.5,
    "keyboard": 45,
    "book": 20,
     "person": 50, 
    "car": 180,
    "motorbike": 200,
    "bus": 350,
    "truck": 400,
    "bicycle": 150,
    "traffic light": 100,
    "stop sign": 75,
    "fire hydrant": 50,
    "chair": 50,
    "diningtable": 150,
    "pottedplant": 30,
    "bed": 150,
    "sofa": 200,
    "toilet": 40,
    "tvmonitor": 100,
    "laptop": 35,
    "microwave": 50,
    "oven": 60,
    "sink": 40,
    "refrigerator": 70,
    "toaster": 30,
    "scissors": 20,
    "teddy bear": 30,
    "hair drier": 20,
    "toothbrush": 20,
    "cup": 10,
    "fork": 20,
    "knife": 20,
    "spoon": 20,
    "bowl": 20,
    "banana": 15,
    "apple": 10,
    "sandwich": 10,
    "orange": 10,
    "broccoli": 20,
    "carrot": 15,
    "hot dog": 15,
    "pizza": 20,
    "donut": 10,
    "cake": 20,
    "backpack": 30,
    "umbrella": 30,
    "handbag": 30,
    "tie": 10,
    "suitcase": 40,
    "frisbee": 20,
    "skis": 150,
    "snowboard": 150,
    "sports ball": 20,
    "kite": 30,
    "baseball bat": 80,
    "baseball glove": 20,
    "skateboard": 80,
    "surfboard": 200,
    "tennis racket": 60,
    "wine glass": 10,
    "bottle": 10,
    "cup": 10,
    "fork": 20,
    "knife": 20,
    "gun": 20,
    "rifle": 100,
    "pistol": 20,
    "scissors": 20,


}


focal_length = 1000  # camera calibration

prev_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1  # Bounding box width

            class_index = int(box.cls[0])
            class_name = classNames[class_index]

            if class_name in object_sizes:
                known_width = object_sizes[class_name]
                distance = (known_width * focal_length) / w  # Distance formula

            # Convert to meters if distance >= 100 cm (1 meter)
            if distance >= 100:
                distance_m = distance / 100
                distance_text = f'{class_name}: {distance_m:.2f} m'
            else:
                distance_text = f'{class_name}: {int(distance)} cm'

            cvzone.putTextRect(img, distance_text, (x1, y1 - 10), scale=1, thickness=1, colorR=(0, 0, 255))

        cvzone.cornerRect(img, (x1, y1, w, y2 - y1), colorR=(0, 255, 0))


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
