import numpy as np
import cv2
import cv2.cv as cv
import json

img = cv2.imread('paper_reduced_shape.png',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cimg_clone = cimg.copy()
cv2.namedWindow('Deviation Gague')

# Globals for Tracking clicks
tracking_centers = []
tracking_labels = ['TL', 'TR', 'BR', 'BL']
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
        selecting = True

    # check to see if the left mouse button was released
    if event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        clicked_tracking_pts.append((x, y))
        cv2.circle(cimg, (x, y) ,5,(255,255,0),3)
        cv2.imshow('Deviation Gague', cimg)
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

    # Add principal points in this order [TL, TR, BR, BL]
    tracking_centers.append(centers[norms_top_left.index(min(norms_top_left))])
    tracking_centers.append(centers[norms_top_right.index(min(norms_top_right))])
    tracking_centers.append(centers[norms_top_left.index(max(norms_top_left))])
    tracking_centers.append(centers[norms_top_right.index(max(norms_top_right))])

    for center in enumerate(tracking_centers):
        # print center
        cv2.circle(cimg,tuple(center[1]),2,(255,0,0),3)
        cv2.putText(cimg, tracking_labels[center[0]], tuple(center[1]), FONT, 1,(255,255,255),2)

def perp_transform():
    """
    Evaluates the use-input points and returns a transformed image.
    """
    # if len(clicked_tracking_pts) != 4:
    #     print "Error: need exactly four clicked points in clockwise order: {}".format(tracking_labels)
    #     return
    # sq_errors = []
    # for truth in enumerate(clicked_tracking_pts):
    #     err = np.linalg.norm(truth[1] - tracking_centers[truth[0]])
    #     sq_errors.append(err ** 2)
    tracking_centers_tup = map(lambda np_arr: tuple(np_arr), tracking_centers)
    # print "Estimated points: {}".format(tracking_centers_tup)
    # print "True (clicked) points: {}".format(clicked_tracking_pts)
    # print "Squared errors: {}".format(sq_errors)

    # Source points are detected tracking points in the iamge
    pts_src = np.array([list(t) for t in tracking_centers_tup]).astype(float)
    # Destination points are the corners of the screen
    pts_dst = np.array([[0 - 10, 0 - 10], [IMG_WIDTH + 10, 0 - 10], [IMG_WIDTH + 10, IMG_HEIGHT + 10], [0 - 10, IMG_HEIGHT + 10]]).astype(float)
    homog, status = cv2.findHomography(pts_src, pts_dst)
    # print homog
    # cimg_copy = cimg.copy()
    return cv2.warpPerspective(cimg, homog, (IMG_WIDTH, IMG_HEIGHT))

def eval_features():
    if len(clicked_tracking_pts) != 4:
        print "Error: need exactly four clicked points in clockwise order: {}".format(tracking_labels)
        return
    sq_errors = []
    for truth in enumerate(clicked_tracking_pts):
        err = np.linalg.norm(truth[1] - tracking_centers[truth[0]])
        sq_errors.append(err ** 2)
    tracking_centers_tup = map(lambda np_arr: tuple(np_arr), tracking_centers)
    print "Estimated points: {}".format(tracking_centers_tup)
    print "True (clicked) points: {}".format(clicked_tracking_pts)
    print "Squared errors: {}".format(sq_errors)

def parse_contour(contours):
    # Form: [[[x1, y1], [x2, y2], ... [xn, yn]] [[x1, y1], ... [xn, yn]]
    data = {}
    data['width'] = 0
    data['height'] = 0
    data['paths'] = []
    for group in contours:
        num_pts = group.shape[0]
        group = group.reshape((num_pts, 2))
        group = group.tolist()
        path = {}
        path['type'] = 'Path'
        path['closed'] = 'true' # NOTE: not sure about this
        path['segments'] = group
        data['paths'].append(path)
    final = { 'data' : data}
    return json.dumps(final)

find_circles()
cv2.imshow('Deviation Gague', cimg)
while True:
    key = cv2.waitKey(1) & 0xFF

    # Threshold image
    if key == ord('t'):
        cimg_grey = cv2.cvtColor(cimg, cv2.COLOR_BGRA2GRAY)
        (thresh, cimg) = cv2.threshold(cimg_grey, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow('Deviation Gague', cimg)

    # Reset user-input tracking points
    if key == ord('r'):
        tracking_centers = []
        clicked_tracking_pts = []
        cimg = cimg_clone.copy()
        find_circles()
        cv2.imshow('Deviation Gague', cimg)

    # Evaluate user-input tracking points
    if key == ord('e'):
        eval_features()

    # Apply a perspective transform and crop
    if key == ord('h'):
        cimg = perp_transform()
        cv2.imshow('Deviation Gague', cimg)

    # Find countours
    if key == ord('c'):
        (contours, hierarchy) = cv2.findContours(cimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = parse_contour(contours)
        with open("contours.json", "w+") as cont_file:
            cont_file.write(str(contours))
            cont_file.flush()
            cont_file.close()
        print "Contours written to contours.json"

    # Quit
    if key == ord ('q'):
        break

cv2.destroyAllWindows()
