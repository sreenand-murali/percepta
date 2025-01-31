from ultralytics import YOLO
import cv2
import logging
from TimedArray import TimedArray
import time

import RPi.GPIO as GPIO


from variables import objects_mapping,class_labels

HAPTIC_PINS = [4, 5, 6, 12, 13, 16, 17, 18, 19, 20, 21, 26]

GPIO.setmode(GPIO.BCM)
GPIO.setup(HAPTIC_PINS, GPIO.OUT)


hapticToPin = {1:4,2:5,3:6,4:12,5:13,6:16,7:17,8:18,9:19,10:20,11:21,12:26}
timed_array = TimedArray(timeout=5)
hapticStack=[]
hapticTimer = 2
hapticTime = time.time()


model = YOLO('yolo11n.pt')  
logging.getLogger('ultralytics').setLevel(logging.ERROR)

cap = cv2.VideoCapture(0)

initialSizes = {}

def objToHaptic(frame):
    global hapticStack
    global timed_array
    global hapticStack
    global hapticTimer
    global hapticTime
    global model  
    global initialSizes
    
    if(time.time()-hapticTime>=hapticTimer):
        GPIO.output(HAPTIC_PINS,GPIO.LOW)


    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy() 

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        
        if conf >= 0.8:
            
            obj_class = int(cls)  
            obj_id = f"{obj_class}" 
            
            name = class_labels[obj_class]["name"]
            
                

            width = x2 - x1
            height = y2 - y1
            size = width * height
            if obj_class >= 0 and obj_class <= 8 and obj_id not in initialSizes:
                initialSizes[obj_id] = size
            if obj_id in initialSizes:
                initialSize = initialSizes[obj_id]
                size_change = size - initialSize

                if(size_change>100000):
                    #/////////////////////////////////////CV2/////////////////////////////
                    # cv2.putText(frame, f"Approaching {class_labels[obj_class]["name"]}", (int(x1), int(y1) - 10),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # hapticStack.append(class_labels[obj_class])
                    #/////////////////////////////////////CV2//////////////////////////////
                    currentObjLabel = class_labels[obj_class]
                    currentObjLabel["approaching"]=1
                    currentObjLabel["priority"]=-2
                    hapticStack.append(currentObjLabel)
                    hapticStack = sorted(hapticStack, key=lambda x: x["priority"])
                    initialSizes.pop(obj_id)
                elif(size_change<-100000):
                #/////////////////////////////////////CV2//////////////////////////////
                    
                    # cv2.putText(frame, f"Going away {class_labels[obj_class]["name"]}", (int(x1), int(y1) - 10),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                #/////////////////////////////////////CV2//////////////////////////////

                    currentObjLabel = class_labels[obj_class]
                    currentObjLabel["approaching"]=-1
                    currentObjLabel["priority"]=-1
                    hapticStack.append(currentObjLabel)
                    hapticStack = sorted(hapticStack, key=lambda x: x["priority"])
                    initialSizes.pop(obj_id)
            #//////////////////Outputting/////////////////////////////
            if name in objects_mapping and obj_class not in timed_array.get_elements():
                timed_array.add(obj_class)
                if(class_labels[obj_class] not in hapticStack):
                    hapticStack.append(class_labels[obj_class])
                    hapticStack = sorted(hapticStack, key=lambda x: x["priority"])
            if(hapticStack and time.time()-hapticTime>=hapticTimer):
                print(hapticStack[0])
                GPIO.output(hapticToPin[objects_mapping[hapticStack[0]["name"]]["left"]], GPIO.HIGH)
                for digit in str(objects_mapping[hapticStack[0]["name"]]["right"]):
                    GPIO.output(hapticToPin[int(digit)+5], GPIO.HIGH)
                hapticStack.pop(0)
                hapticTime=time.time()
            #//////////////////Outputting/////////////////////////////

            
            #///////////////////cv2//////////////////////////////
            
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            
            # label = f"{class_labels[obj_class]["name"]}: {conf:.2f}"  
            # label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            # label_x = int(x1)
            # label_y = int(y1) - 10
            # cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #///////////////////cv2//////////////////////////////



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    objToHaptic(frame)
                
    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('f'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
