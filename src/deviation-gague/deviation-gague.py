import numpy as np
import cv2
import cv2.cv as cv

img = cv2.imread('paper_reduced.jpg',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

# Window setup
def nothing(x):
    pass

cv2.namedWindow('detected circles')
cv2.createTrackbar('min_dist','detected circles',10,100,nothing)
cv2.createTrackbar('dp','detected circles',1,5,nothing)

while True:
    min_dist = cv2.getTrackbarPos('min_dist', 'detected circles')
    dp = cv2.getTrackbarPos('dp', 'detected circles')
    circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,1,10,
                               param1=50,param2=30,minRadius=0,maxRadius=0)

    if circles is None:
        pass
    else:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
           # draw the outer circle
           cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
           # draw the center of the circle
           cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',cimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
