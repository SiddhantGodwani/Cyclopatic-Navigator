# pip install opencv-contrib-python = Computer vision software
# pip install mediapipe             = Detects eye movements
# pip install pyautogui             = Auto GUI interface for python (Easy to use)

#------------------------------------------------------------------------------------------
#Step 1 = Openingh webcam and seeing our face

import cv2         #for S1
import mediapipe as mp   #for S2


cam = cv2.VideoCapture(0)  # Captures video from camera aat index 0

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)   #for S2



# for video to run every framme after frame
while True:   #runs forever
    _ , frame = cam.read()      #read every frame(image/video)
    
    # converts input from frame to black and white for easy detection#
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #for S2
    
    output = face_mesh.process(rgb_frame)  #process the output image in frame  #for S2
    landmarks_points = output.multi_face_landmarks  #detect landmark points and return them to output   #for S2
    #print(landmarks_points) #for S2-->commented as we now know that landmarks are detected
    
    
    #Detect frames height and width to multiply with x , y axes to get centre values
    frame_h, frame_w , _ =  frame.shape
    
    if landmarks_points:
        landmarks= landmarks_points[0].landmark  #Detect landmarks of only one face(ie at index 0)
        
        #looping through landmark points and showing them
        for landmark in landmarks[474:478]:   # detect refined land marks of eyes ie. 474 to 478
            x = int(landmark.x * frame_w)  #multiplied and converted to int for making circle    
            y = int(landmark.y * frame_h)
            cv2.circle(frame,(x,y), 3 , (0,255,0))  #draw circle - in frame-> at centre x,y ->of size 3 -> of color RGB
            print(x,y)
            
            
    
    
    cv2.imshow("Hand Less Mouse" , frame)  #--> show image
    cv2.waitKey(1)              # waits for a key and then 1 second -> cexecutes
    
    
    
#Step 2 =  Detecting face and eye
#Step 3 =  Show face and landmark points(ie eyes and face)



