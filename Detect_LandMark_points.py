# pip install opencv-contrib-python = Computer vision software
# pip install mediapipe             = Detects eye movements
# pip install pyautogui             = Auto GUI interface for python (Easy to use)

#------------------------------------------------------------------------------------------
#Step 1 = Openingh webcam and seeing our face

import cv2         #for S1
import mediapipe as mp   #for S2


cam = cv2.VideoCapture(0)  # Capyures video from camera aat index 0

face_mesh = mp.solutions.face_mesh.FaceMesh()   #for S2



# for video to run every framme after frame
while True:   #runs forever
    _ , frame = cam.read()      #read every frame(image/video)
    
    # converts input from frame to black and white for easy detection#
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #for S2
    
    output = face_mesh.process(rgb_frame)  #process the output image in frame  #for S2
    landmarks_points = output.multi_face_landmarks  #detect landmark points and return them to output   #for S2
    print(landmarks_points) #for S2
    
    cv2.imshow("Hand Less Mouse" , frame)  #--> show image
    cv2.waitKey(1)              # waits for a key and then 1 second -> cexecutes
    
#Step 2 =  Detecting face and eye


