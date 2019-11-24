# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:00:37 2019

@author: Dell-pc
"""

from flask import Flask
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


app = Flask(__name__)

@app.route('/')
def hello_world():
    
    def sound_alarm(path):
	    playsound.playsound(path)

    def eye_aspect_ratio(eye):
    	A = dist.euclidean(eye[1],eye[5])
    	B = dist.euclidean(eye[2],eye[4])
    
    	C = dist.euclidean(eye[0],eye[3])
    
    	ear = (A + B)/(2.0 * C)
    
    	return ear
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", default = "shape_predictor_68_face_landmarks.dat", help = "path to facial landmark predictor")
    ap.add_argument("-a", "--alarm", type = str, default = "alarm.wav", help = "path alarm .WAV file")
    ap.add_argument("-w", "--webcam", type = int, default = 0, help = "index of webcam on system")
    args = vars(ap.parse_args())
    
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 48
    
    Counter = 0
    alarm_on = False
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    
    (lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    print("Starting stream thread...")
    vs = VideoStream(src = args["webcam"]).start()
    time.sleep(1.0)
    
    while True:
    	frame = vs.read()
    	frame = imutils.resize(frame, width = 450)
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    	rects = detector(gray,0)
    
    	for rect in rects:
    		shape = predictor(gray,rect)
    		shape = face_utils.shape_to_np(shape)
    
    		left_eye = shape[lstart:lend]
    		right_eye = shape[rstart:rend]
    		leftEAR = eye_aspect_ratio(left_eye)
    		rightEAR = eye_aspect_ratio(right_eye)
    
    		ear = (leftEAR + rightEAR)/ 2.0
    
    		leftEyeHull = cv2.convexHull(left_eye)
    		rightEyeHull = cv2.convexHull(right_eye)
    		cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0),1)
    		cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0),1)
    
    		if ear<EYE_AR_THRESH:
    			Counter +=1
    
    			if Counter>=EYE_AR_CONSEC_FRAMES:
                    
                   # sum = (sum+ear)
                   # flag = flag + 1
    
    				if not alarm_on:
    					alarm_on = True
    
    					if args["alarm"] != "":
    						t = Thread(target= sound_alarm, args = (args["alarm"],))
    						t.deamon = True
    						t.start()
    
    				cv2.putText(frame,"DROWSINESS ALERT!!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
                    sum = (sum+ear)
    		else:
    			Counter = 0
    			alarm_on = False
    
    		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
     	
    	cv2.imshow("Frame", frame)
    	key = cv2.waitKey(1) & 0xFF
    
    	if key == ord("q"):
    		break
    
    cv2.destroyAllWindows()
    vs.stop()
    
    return "Ride Ended"

if __name__ == '__main__':
    app.run()
