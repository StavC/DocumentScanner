import numpy as np
import cv2

def points(pts):
    #will hold the points in order top-left, top-right, bottom-left,bottom-right
    rect=np.zeros((4,2), dtype='float32')

    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[3]=pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]

    return rect


def four_points_transform(image,pts):
    rect=points(pts)

    (top_left,top_right,bottom_left,bottom_right)=rect
    #calculate the ecu-distance between top and bot to get the max width of the object
    width_top=np.sqrt(((top_right[0]-top_left[0])**2)+((top_right[1]-top_left[1])**2))
    width_bot=np.sqrt(((bottom_right[0]-bottom_left[0])**2)+((bottom_right[1]-bottom_left[1])**2))
    max_width=max(int(width_top),int(width_bot))

    height_top = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height_bot = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    max_height= max(int(height_top), int(height_bot))

    object_dim=np.array([
        [0,0],[max_width-1,0],
        [0,max_height-1],[max_width-1,max_height-1]]
        ,dtype='float32')

    matrix=cv2.getPerspectiveTransform(rect,object_dim)
    pic=cv2.warpPerspective(image,matrix,(max_width,max_height))

    return pic




