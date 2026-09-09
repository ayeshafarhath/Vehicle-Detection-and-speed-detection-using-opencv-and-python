import numpy as np
import cv2
import time
car_cascade = cv2.CascadeClassifier('cars.xml')
cap = cv2.VideoCapture('car.mp4')
wide=0.1  
flag=True
start=end=0
time_diff=0
while(cap.isOpened()):
    ret, img = cap.read()
    height,width,chan=img.shape   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in cars:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
        center_x=(2*x+w)/2
        center_y=(2*y+h)/2 
        dist1=((wide*668.748634441)/w)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        dist0=((wide*668.748634441)/w)
        actual_dist=dist0*(width/2)/668.748634441      
        if flag is True and int(round(center_x)) in (range(0,80) or range(400,480)):
            start=time.time()
            flag=False
        if flag is False and int(round(center_x)) in range(int(round(width/2))-10,int(round(width/2))+10):
            end=time.time()
            time_diff=end-start           
            flag=True
            s_flag=True   
    if time_diff>0 and s_flag==True:
        velocity=actual_dist/time_diff       
        vel_kmph=round(velocity*3.6,2)
        print("Speed:",vel_kmph,"kmph")
        print("Distance from car:",round(dist1,2),"m")
        s_flag=False
    cv2.line(img,(int(width/2),0),(int(width/2),height),(255,0,0),2)        
    cv2.imshow('frame',img) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
