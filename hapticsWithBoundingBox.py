from ultralytics import YOLO
import cv2
from deepface import DeepFace
import logging
from TimedArray import TimedArray
import time
import requests

from variables import objects_mapping, class_labels

LEFT_NODEMCU_IP = "http://192.168.20.19"
RIGHT_NODEMCU_IP = "http://192.168.20.3"

HAPTIC_PINS = [4, 5, 6, 12, 13, 16, 17, 18, 19, 20, 21, 26]
hapticToPin = {1: 4, 2: 5, 3: 6, 4: 12, 5: 13, 6: 16, 7: 17, 8: 18, 9: 19, 10: 20, 11: 21, 12: 26}

timed_array = TimedArray(timeout=5)
hapticStack = []
hapticTimer = 2
hapticTime = time.time()

model = YOLO('yolo11n.pt')
logging.getLogger('ultralytics').setLevel(logging.ERROR)

cap = cv2.VideoCapture(0)
initialSizes = {}

def send_haptic_signal(left_motor, right_motor):
    try:
  
        requests.get(f"{LEFT_NODEMCU_IP}/gpio?command=1{left_motor}")
        requests.get(f"{RIGHT_NODEMCU_IP}/gpio?command=1{right_motor}")
        print(f"{LEFT_NODEMCU_IP}/gpio?command=1{left_motor}")
        print(f"{RIGHT_NODEMCU_IP}/gpio?command=1{right_motor}")

    except Exception as e:
        print("Error sending haptic signal:", e)

def objToHaptic(frame):
    global hapticStack, timed_array, hapticTimer, hapticTime, model, initialSizes

    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf >= 0.8:
            obj_class = int(cls)
            obj_id = f"{obj_class}"
            name = class_labels[obj_class]["name"]

            # --- Bounding Box and Annotation ---
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{name} ({conf:.2f})"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

            # --- Haptic Logic ---
            width = x2 - x1
            height = y2 - y1
            size = width * height
            
           
            if obj_class >= 0 and obj_class <= 8 and obj_id not in initialSizes:
                initialSizes[obj_id] = size
            if obj_id in initialSizes:
                initialSize = initialSizes[obj_id]
                size_change = size - initialSize
                if size_change > 100000:
                    currentObjLabel = class_labels[obj_class].copy()
                    currentObjLabel["approaching"] = 1
                    currentObjLabel["priority"] = -2
                    hapticStack.append(currentObjLabel)
                    hapticStack = sorted(hapticStack, key=lambda x: x["priority"])
                    initialSizes.pop(obj_id)
                elif size_change < -100000:
                    currentObjLabel = class_labels[obj_class]
                    currentObjLabel["approaching"] = -1
                    currentObjLabel["priority"] = -1
                    hapticStack.append(currentObjLabel)
                    hapticStack = sorted(hapticStack, key=lambda x: x["priority"])
                    initialSizes.pop(obj_id)

            if name in objects_mapping and obj_class not in timed_array.get_elements():
                timed_array.add(obj_class)
                if class_labels[obj_class] not in hapticStack:
                    hapticStack.append(class_labels[obj_class])
                    if obj_class == 0:
                        emotion_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                        dominant_emotion = emotion_result[0]['dominant_emotion']
                        if dominant_emotion != "neutral":
                            hapticStack.append({
                                "name": dominant_emotion,
                                "priority": class_labels[obj_class]["priority"] + 0.5
                            })

                            # Annotate emotion on frame
                            cv2.putText(frame, f"Emotion: {dominant_emotion}", (int(x1), int(y2) + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    hapticStack = sorted(hapticStack, key=lambda x: x["priority"])

            if hapticStack and time.time() - hapticTime >= hapticTimer:
                print(hapticStack)
                if hapticStack[0].get("approaching", 0) == 1:
                    left_motor = objects_mapping[hapticStack[0]["name"]]["left"]*10+5
                else:
                    left_motor = objects_mapping[hapticStack[0]["name"]]["left"]
                right_motor = objects_mapping[hapticStack[0]["name"]]["right"]
                send_haptic_signal(left_motor, right_motor)
                hapticStack.pop(0)
                hapticTime = time.time()

# --- Main loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    objToHaptic(frame)
    cv2.imshow("YOLOv8 Detection with Haptics", frame)

    if cv2.waitKey(1) & 0xFF == ord('f'):
        break

cap.release()
cv2.destroyAllWindows()
