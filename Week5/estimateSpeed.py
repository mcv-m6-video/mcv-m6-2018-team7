from __future__ import division
import numpy as np
import cv2


def computeRectificationHomography(image):

    # Get manually 4 points that form 2 parallel lines
    points = getXY(image)

    p1 = np.array([points[0,0], points[0,1], 1])
    p2 = np.array([points[1,0], points[1,1], 1])
    p3 = np.array([points[2,0], points[2,1], 1])
    p4 = np.array([points[3,0], points[3,1], 1])
    p5 = np.array([points[4,0], points[4,1]])
    p6 = np.array([points[5,0], points[5,1]])

    l1 = np.cross(p1, p2)
    l2 = np.cross(p3, p4)

    # Compute the vanishing point
    v = np.cross(l1, l2)
    v = v/v[2]

    dist_Ref = 1
    # Create homography for aereal view
    H = np.array([[1, -v[0]/v[1], 0],
                 [0,           1, 0],
                 [0,     -1/v[1], 1]])

    # for highway dataset (computed in Matlab)
    # H = np.array([[0.5820, 0.6971, -110.0020], [-0.0088, 0.9621, -57.5384], [-0.0001, 0.0025, 0.1978]])

    # for video2 dataset (computed in Matlab)
    # H = np.array([[-0.7673, -0.3906, 67.8345], [-0.0610, -0.8039, 35.1424], [-0.0006, -0.0022, -0.2235]])

    H = H /H[2,2]
    dist = computeDistanceInPixels(p5[::-1], p6[::-1], H)
    pixelToMeters = dist_Ref / dist

    im_warped = cv2.warpPerspective(image, H, (image.shape[1],image.shape[0]))
    cv2.imshow('Result', im_warped)
    cv2.waitKey(0)

    return H, pixelToMeters


def computeDistanceInPixels(p1, p2, H):

    # Convert points to homogeneous coordinates
    p1_hom_cord = np.array([p1[0], p1[1], 1])
    p2_hom_cord = np.array([p2[0], p2[1], 1])

    # Rectify points by apllying Homogragy
    p1_rect = np.dot(H,np.transpose(p1_hom_cord))
    p2_rect = np.dot(H,np.transpose(p2_hom_cord))

    # Normalize homogeneous coordinates
    p1_hom_cord = p1_rect / p1_rect[2]
    p2_hom_cord = p2_rect / p2_rect[2]

    # Convert to cartesian coordinates
    p1_cart = p1_hom_cord[:2]
    p2_cart = p2_hom_cord[:2]

    # Compute distance (diference in vertical direction)
    dist = p2_cart[0]-p1_cart[0]
    return dist


def estimate_speed(actual_centroid, actual_frame_numb,  prev_centroid, prev_frame_number ,  H, fps, PixelToMeters):

    # Compute distance in pixels between centroids
    dist_pix = computeDistanceInPixels(prev_centroid, actual_centroid, H)

    # Compute distnce in meters
    dist_met = dist_pix * PixelToMeters
    dt = (actual_frame_numb-prev_frame_number)/fps
    speed = ((dist_met / dt)*3600)/1000

    return speed


a = np.array([0, 0], dtype='float32')


def getXY(image):
    global a

    # Set mouse CallBack event
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', getxy)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    b = a[1:,:].copy()

    return b


# define the event
def getxy(event, x, y, flags, param):
    global a

    if event == cv2.EVENT_LBUTTONDOWN:
        a = np.vstack([a, np.hstack([x, y])])
        print "(col, row) = ", (x, y)
    return