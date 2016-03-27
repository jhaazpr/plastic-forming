import numpy as np
import cv2
import cv2.cv as cv

img = cv2.imread('paper_reduced.png',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

IMAGE_DIMS = img.shape
IMG_WIDTH = IMAGE_DIMS[1]
IMG_HEIGHT = IMAGE_DIMS[0]
TOP_LEFT = np.array([0, 0])
TOP_RIGHT = np.array([0, IMG_WIDTH])

circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,2,20,
                           param1=50,param2=30,minRadius=0,maxRadius=40)


if circles is None:
    pass
else:
    circles = np.uint16(np.around(circles))
    # centers = circles[:, :, 0:2][0]
    # print centers
    # print np.linalg.norm(centers[0] - TOP_LEFT)

    norms_top_left = []
    norms_top_right = []
    centers = []
    principal_centers = []

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
    principal_centers.append(centers[norms_top_left.index(min(norms_top_left))])
    principal_centers.append(centers[norms_top_left.index(max(norms_top_left))])
    principal_centers.append(centers[norms_top_right.index(min(norms_top_right))])
    principal_centers.append(centers[norms_top_right.index(max(norms_top_right))])

    print principal_centers

    for center in principal_centers:
        cv2.circle(cimg,tuple(center),2,(255,0,0),3)

cv2.imshow('detected circles',cimg)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


# raw
# processing (cleaning)
# feature extraction (get the four points)
# evaluation (I click the true points, this thing compares the distance)
