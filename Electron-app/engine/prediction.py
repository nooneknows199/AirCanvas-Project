from typing import Counter
import cv2 as cv
import mediapipe as mp
from mediapipe.python.packet_creator import create_image_frame
import numpy as np
import math as m
from sklearn.metrics import pairwise
import os
from keras.models import model_from_json
import operator
import pyautogui as pg
# import matplotlib as plt

current_file_path = os.path.dirname(os.path.realpath(__file__))
directory = current_file_path
os.chdir(directory)

mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 1, 0.65, 0.5)


# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

# Category dictionary
# categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}

canvas = None
x1,y1 = 0,0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv.VideoCapture(0)

# frame_count = 0

# prev_frame = img


def lmList(img, handNo = 0):
    lmLst = []
    results = hands.process(img)

    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print(id, cx, cy)
            lmLst.append([id, cx, cy])

    return lmLst

def drawHands(img):
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    

    if results.multi_hand_landmarks:
        
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)        
    return img


def main():

    global canvas,x1,y1
    cap = cv.VideoCapture(0)
    frame = cap
    # For FPS
    prevTime = 0
    curTime = 0
    
    

    while True:
        success, img = cap.read()

        if not success:
            print("Please check your webcam.")
            continue

        if canvas is None:
            canvas = np.zeros_like(img)

        img = cv.flip(img, 1)
        img = drawHands(img)

        lmLst = lmList(img)

        height, width = img.shape[:2]
        
        if height == 0 or width == 0:
            continue

        if len(lmLst) != 0:

                
            ix, iy = lmLst[8][1], lmLst[8][2]

            # Area of hand:
            # Middle Finger (Top)
            #mx, my = lmLst[8][1], lmLst[8][2]
            mx, my = ix, iy
            # Wrist (Bottom)
            wx, wy = lmLst[0][1] , lmLst[0][2]

            # Pinky finger (right)
            px, py = lmLst[20][1], lmLst[20][2]

            # Thumb (left) 
            tx , ty = lmLst[4][1], lmLst[4][2]

            fingers = []
            tipIds = [8, 12, 16, 20]

            # # Thumb
            # if lmLst[tipIds[0]][1] > lmLst[tipIds[0] - 1][1]:
            #     fingers.append(1)
            # else:
            #     fingers.append(0)

            # 4 Fingers
            for id in range(0, 4):
                if lmLst[tipIds[id]][2] < lmLst[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Fingers Opened

            # print(fingers)

            if fingers[1] == 1 :   # Middle Finger
                mx, my = lmLst[12][1], lmLst[12][2]
            elif fingers[0] == 1 : # Index Finger
                mx, my = lmLst[8][1], lmLst[8][2]
            elif fingers[2] == 1 : # Ring Finger
                mx, my = lmLst[16][1], lmLst[16][2]
            elif fingers[3] == 1 : # Pinky Finger
                mx, my = lmLst[20][1], lmLst[20][2]


            top    = (mx,my)
            bottom = (wx,wy)
            right  = (px,py)
            left   = (tx,ty)

            # In theory, the center of the hand is half way between the top and bottom and halfway between left and right
            cX = (left[0] + right[0]) // 2
            cY = (top[1] + bottom[1]) // 2
            
            distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
            
            # Grab the largest distance
            max_distance = distance.max()
            
            # Create a circle with 1% radius of the max euclidean distance
            radius = int(max_distance)

            # Now grab an ROI of only that circle
            #circular_roi = np.zeros(img.shape, dtype="uint8")

            x1, y1 = (cX - radius), (cY + radius)
            x2, y2 = (cX + radius), (cY - radius) 
            
            roi = img[y2:y1,x1:x2]
            height, width = roi.shape[:2]
        
            if height <= 0 or width <= 0:
                continue
            roi = cv.resize(roi, (128,128))


            print("shape ",roi.shape)

            cv.imshow("ROI", roi)

            roi = cv.resize(roi, (64, 64)) 
            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

            thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)[1]

            fgmask = cv.morphologyEx(thresh, cv.MORPH_OPEN, (3,3), iterations=4)
            # To close the holes in the objects
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, (5,5), iterations=5)
            # cv.imshow("test", test_image)

            result = loaded_model.predict(fgmask.reshape(1, 64, 64, 1))
            # print(type(result))
            print("result", result)

                # {'Lshape': 0, 'broFist': 1, 'fiveF': 2, 'okay': 3, 'palm': 4, 'twoF': 5}

            prediction = {'twoF': result[0][2],
                        'broFist': result[0][0],
                        'palm': result[0][1],}
                        
            # Do whatever u wanna do here
            if prediction.get('Lshape'):
                cv.putText(img,f"L-shaped hand",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)

            if prediction.get('broFist'):
                cv.putText(img,f"Bro Fist",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)

            if prediction.get('fiveF'):
                cv.putText(img,f"Playing current track",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)
                #pg.press("playpause")

            if prediction.get('okay'):
                cv.putText(img,f"Okay",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)

            if prediction.get('palm'):
                cv.putText(img,f"Pausing current track",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)
                #pg.press("pause")


            if prediction.get('twoF'):
                cv.putText(img,f"Two Fingers",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)
           



            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            
            cv.putText(img, prediction[0][0], (10, 120), cv.FONT_HERSHEY_PLAIN, 2, (31,240,31), 2)    
            cv.imshow("img", img)
            # img = cv.add(img,canvas)
            # stacked = np.hstack((frame,img))
            # cv.imshow('Beautiful Gestures', cv.resize(stacked,None,fx=1.2,fy=1.2))
                
                
        # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = cv.add(img,canvas) #to see the result in img too 
        '''
        This ^^ sometimes causes error:
        error: (-209:Sizes of input arguments do not 
        match) The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array' in function 'cv::arithm_op'
        '''
        stacked = np.hstack((img,canvas))
        cv.imshow('Beautiful Gestures', cv.resize(stacked,None,fx=1.2,fy=1.2))
        

        # frame_count += 1
        # prev_frame = img.copy()
        # success, img = cap.read() # img = crurent img

        key = cv.waitKey(1) & 0xFF
        if key == ord('d'):
            # print(f"Frames found: {frame_count}")
            break
        if key == ord('c'): #clear canvas
            canvas = None


    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()