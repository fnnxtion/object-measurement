#import library yang dibutuhkan (OpenCV, Numpy, Math, Time, Tracker)
import cv2
from cv2 import aruco
import numpy as np
import math 
import time
# import serial
from tracker4 import *
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
tracker = Tracker()

#Menghubungkan dengan port Arduino di COM5 baudrate 9600
ser = serial.Serial('COM5', 9600)

#inisialisasi variabel yang dibutuhkan
detected_object = 0 
frame_index = 0
apple_count = 0
defect_count = 0
class1_count = 0
class2_count = 0
class3_count = 0
ge=0
gb=0
gc=0
gd = 0
size = 0
obj_id = None
previous_ids = set()
prev_frame_time = 0
new_frame_time = 0
object_count = 0

#mengambil input berupa video/real-time
cap = cv2.VideoCapture('measure/d2.mp4')
# cap = cv2.VideoCapture('http:192.168.100.18:4747/video')

#load model yang telah di build
net = cv2.dnn.readNetFromONNX("v3.onnx")
file = open("classes.txt","r")
classes = file.read().split('\n')
print(classes)

#ROI yang disesuaikan untuk real-time/video
polygon_points = np.array([[60, 607], [60, 580], [641, 580], [641, 607]])

#realtime used
#polygon_points = np.array([[520, 287], [520, 586],[750, 586] , [750, 287]])
#polygon_points2 = np.array([[350, 287], [350, 586],[478, 586] , [478, 287]])

#loop untuk menampilkan setiap frame
while True:
    #memastikan ada input
    ret, frame = cap.read()
    if frame is None:
        break
    # img = cv2.resize(img, (640,640))

    #resize frame untuk penyesuaian UI
    height, width = frame.shape[:2]
    crop_size = min(height, width)
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size
    img = frame[start_y:end_y, start_x:end_x]
    img = cv2.resize(img, (1000,800))

    #menampilan fps dan line ROI di UI
    polylines_layer = np.zeros_like(img)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(img, "fps: " + fps, (610, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.polylines(polylines_layer, [polygon_points], True, (0, 255, 255, 0), 2)  # Draw polyline with transparent yellow color
    #cv2.polylines(polylines_layer, [polygon_points2], True, (0, 255, 255, 0), 2)

    #deteksi aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if corners:
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 2)

        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)
        pixel_cm_ratio = aruco_perimeter / 200

        #menjalankan model pada input
        blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
        net.setInput(blob)
        detections = net.forward()[0]
        classes_ids = []
        confidences = []
        boxes = []
        labels = [] 
        objects_rect = [] 
        rows = detections.shape[0]
        img_width, img_height = img.shape[1], img.shape[0]
        x_scale = img_width/640
        y_scale = img_height/640

        #output deteksi ditampilkan jika confidence score>0.7
        for i in range(rows):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.7:
                classes_score = row[5:]
                ind = np.argmax(classes_score)
                if classes_score[ind] > 0.5:
                    classes_ids.append(ind)
                    confidences.append(confidence)
                    labels.append(classes[ind])
                    x, y, w, h = row[:4]
                    x1 = int((x- w/2)*x_scale)
                    y1 = int((y-h/2)*y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1,y1,width,height])
                    objects_rect.append([x1, y1, width, height])
                    boxes.append(box)

                    #menghitung diameter objek
                    obj_width = w / pixel_cm_ratio
                    obj_height = h / pixel_cm_ratio
        #Non Maximum Supression            
        indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.8)

        for i in indices:
            #menjadikan output bounding box menjadi 1 variabel
            x1,y1,w,h = boxes[i]
            conf = confidences[i]

            #update tracker dengan input bounding box, label dan confidence score
            boxes_ids = tracker.update(objects_rect,  labels, confidences)
            for (x1, y1, width, height,object_id, label, confidence) in boxes_ids:
                id = object_id
                cv2.putText(img, "LB: "+str(label), (x1+250,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, "ID: "+str(id), (x1+150,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

            #menentukan titik tengah objek 
            object_center = (int(x1 + w //2), int(y1+h // 2)) 

            #menentukan apakah objek berada di dalam ROI
            is_inside = cv2.pointPolygonTest(np.array(polygon_points,np.int32), object_center, False)
            is_inside2 = cv2.pointPolygonTest(np.array(polygon_points,np.int32), object_center, False)
            obj_id = id

            #jika objek berada di dalam ROI
            if is_inside == 1:
                #cek ID objek
                if obj_id not in previous_ids:
                    #klasifikasi objek menjadi Grade B, Grade C, Grade D, dan Grade E
                    sizee = min(obj_width, obj_height)
                    size = float(round(sizee, 1))

                    #Grade E
                    if label == 'defect':
                        defect_count += 1
                        ge = defect_count
                        detected_object = 4
                        message = str(detected_object) + "\n"
                        ser.write(message.encode()) 
                        print("apple decfect"+str(defect_count))

                    #Grade D
                    elif size <= 60:
                        class3_count += 1
                        gd = class3_count
                        detected_object = 1
                        message = str(detected_object) + "\n"
                        ser.write(message.encode()) 
                        print("apple class 1"+str(class3_count))

                    #Grade C
                    elif 61 < size < 72: 
                        class2_count += 1
                        gc = class2_count
                        detected_object = 2
                        message = str(detected_object) + "\n"
                        ser.write(message.encode()) 
                        print("apple class 2"+str(class2_count))

                    #Grade B
                    elif size >= 73:
                        class1_count += 1
                        gb = class1_count
                        detected_object = 3
                        message = str(detected_object) + "\n"
                        ser.write(message.encode()) 
                        print("apple class 3"+str(class1_count))

                    previous_ids.add(obj_id)

            cv2.putText(img, "class_1: "+str(class1_count), (610, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, "class_2: :"+str(class2_count), (610, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, "class_3: "+str(class3_count), (610, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, "defect: "+str(defect_count), (610, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, "size: "+str(size), (610, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, "Width {} mm".format(round(obj_width, 1)), (int(x1 - 100), int(y1 - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv2.putText(img, "Height {} mm".format(round(obj_height, 1)), (int(x1 - 100), int(y1 + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    
    #overlay garis ROI sehingga tidak menimbulkan bias pada saat deteksi
    overlay = cv2.addWeighted(img, 1, polylines_layer, 0.5, 0)

    #menampilkan output program
    cv2.imshow("VIDEO",overlay)
    frame_index += 1
    cv2.setMouseCallback("VIDEO", mouse_callback)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()