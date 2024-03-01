import cv2
import math
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import cvzone


from ultralytics import YOLO
from sort import *


def checkDevice():
    # Test cuda availability
    try:
        torch.cuda.is_available()
    except:
        device = 'cpu'
    else:
        device = 'cuda:0'
    finally:
        print('Running on %s' % device)
        return device
    
def checkVideo(videoPath):
    if not os.path.exists(videoPath):
        print('Video not found')
        exit()
    else:
        video = cv2.VideoCapture(videoPath)
        return video

def draw_boxes(img, className, pred, color=(255, 0, 255)): # deal with each frame

    current_frame_detection = np.empty((0,5)) # get current frame all box coordinates

    for result in pred:
        for box in result.boxes: # deal with each bounding box
            # Get the coordinates of the box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convert to int
            w, h = x2 - x1, y2 - y1
            # Get the confidence score
            conf = math.ceil(box.conf[0] * 100) / 100
            # Get the predicted class label
            cls = className[int(box.cls[0])]

            if (cls == 'car' or cls == 'truck' or cls == 'bus') and conf > 0.3:
                # Draw the box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                # Draw the label
                # cv2.putText(img, '%s %.2f' % (cls, conf), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                current_frame_detection = np.vstack((current_frame_detection, np.array([x1, y1, x2, y2, conf]))) # vertical stack => mtx
                
    return img, current_frame_detection
    


def main(videoPath, modelName):
    device = checkDevice()  # Check device for running the model
    #model = YOLO(modelName).to(device)  # Load model
    model = YOLO(modelName)  # Load model
    video = checkVideo(videoPath)  # Load video
    classes = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]  # class list for COCO dataset
    
    # mask
    mask = cv2.imread("mask.png")
    mask = cv2.resize(mask, (1280, 720))
    # print(mask.shape)
    # cv2.imwrite('mask_after_resize.jpg', mask)

    # car icon
    car = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)

    # line
    line = [320, 350, 690, 350]

    # total count list
    total_count_list = []
    

    # Tracking: tracker initial
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


    # Loop
    while True:
        success, frame = video.read()  # Read frame
        if not success:
            break
        frame = cvzone.overlayPNG(frame, car, (0,0)) # overlap car icon
        #print(frame.shape)


        # Detect
        results = model(cv2.bitwise_and(frame,mask),verbose=False) # result list of detections

        # Draw
        frame, detection = draw_boxes(frame, classes, results, color=(255, 245, 152)) # draw box
        
        # tracker update deal with each frame
        results_tracker = tracker.update(detection) # use current frame all box coordinates to update
        for result_tracker in results_tracker:
            x1, y1, x2, y2, id = result_tracker
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id) # Convert to int
            #print(result_tracker)
            
            cv2.putText(frame, '%s' % id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2) # text id

            center = ( (x2+x1)//2, (y2+y1)//2 )
            # cv2.circle(frame, center, 5, (0,0,255), 2)
            if line[0] < center[0] < line[2] and line[1]-20 < center[1] < line[1]+20:
                if id not in total_count_list:
                    total_count_list.append(id)


        cv2.line(frame, (line[0],line[1]), (line[2], line[3]), (0,0,255), thickness=3) # draw limit line
        cv2.putText(frame, '%s' % len(total_count_list), (240, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 215, 255), 7) # draw counter text

        # Show
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # close all windows
    cv2.destroyAllWindows()
    os.system('clear')


if __name__ == '__main__': 
    videoPath = 'cars.mp4'
    modelName = 'yolov8n.pt'
    main(videoPath, modelName)