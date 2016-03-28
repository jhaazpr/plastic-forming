import numpy as np
import cv2
import cv2.cv as cv

img = cv2.imread('paper_reduced.png',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
cimg_clone = cimg.copy()
cv2.namedWindow('Deviation Gague')

# Messy global overhead for click callback
clicked_tracking_pts = []
selecting = False

IMAGE_DIMS = img.shape
IMG_WIDTH = IMAGE_DIMS[1]
IMG_HEIGHT = IMAGE_DIMS[0]
TOP_LEFT = np.array([0, 0])
TOP_RIGHT = np.array([IMG_WIDTH, 0])
FONT = cv2.FONT_HERSHEY_SIMPLEX

def click_set_tracking_point(event, x, y, flags, param):
    # grab references to globals
    global clicked_tracking_pts, selecting

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_tracking_pts = [(x, y)]
        selecting = True
        cv2.circle(cimg, (x, y) ,2,(255,0,0),3)
        cv2.imshow('Deviation Gague', cimg)

    # check to see if the left mouse button was released
    if event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        clicked_tracking_pts.append((x, y))
        selecting = False

        # draw a rectangle around the region of interest
        #cv2.circle(cimg, (x, y) ,2,(255,0,0),3)
        #cv2.imshow('Deviation Gague', cimg)

cv2.setMouseCallback('Deviation Gague', click_set_tracking_point)

def find_circles():
    circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,2,20,
                           param1=50,param2=30,minRadius=0,maxRadius=40)

    if circles is None:
        return

    circles = np.uint16(np.around(circles))
    # centers = circles[:, :, 0:2][0]
    # print centers
    # print np.linalg.norm(centers[0] - TOP_LEFT)

    norms_top_left = []
    norms_top_right = []
    centers = []
    tracking_centers = []
    tracking_labels = ['TL', 'BR', 'TR', 'BL']

    for i in circles[0,:]:
        # draw the outer circle
        center = np.array([i[0], i[1]])
        center_tup = (i[0], i[1])
        radius = i[2]
        cv2.circle(cimg,center_tup,radius,(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,center_tup,2,(0,0,255),3)
        # compute distances from the top left and top right points
        centers.append(center)
        norms_top_left.append(np.linalg.norm((i[0],i[1]) - TOP_LEFT))
        norms_top_right.append(np.linalg.norm((i[0],i[1]) - TOP_RIGHT))

    # Add principal points in this order [TL, BR, TR, BL]
    tracking_centers.append(centers[norms_top_left.index(min(norms_top_left))])
    tracking_centers.append(centers[norms_top_left.index(max(norms_top_left))])
    tracking_centers.append(centers[norms_top_right.index(min(norms_top_right))])
    tracking_centers.append(centers[norms_top_right.index(max(norms_top_right))])

    for center in enumerate(tracking_centers):
        print center
        cv2.circle(cimg,tuple(center[1]),2,(255,0,0),3)
        cv2.putText(cimg, tracking_labels[center[0]], tuple(center[1]), FONT, 1,(255,255,255),2)

find_circles()
cv2.imshow('Deviation Gague', cimg)
while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        clicked_tracking_pts = []
        cimg = cimg_clone
        # find_circles()
        cv2.imshow('Deviation Gague', cimg)

    if key == ord('e'):
        pass

    if key == ord ('q'):
        break

cv2.destroyAllWindows()
