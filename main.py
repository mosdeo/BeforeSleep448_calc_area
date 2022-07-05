import cv2 as cv
import numpy as np

img = cv.imread('BeforeSleep448_2030_screenshot.png')

male_color = img[800, 1300]
male_gray = img[700, 1300]

female_color = img[800, 1500]
female_gray = img[700, 1500]

b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

def color_mask(img, bgr_color):
    err = 35
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    return \
        np.where(bgr_color[0]+err > b, 1, 0) & np.where(bgr_color[0]-err < b, 1, 0) &\
        np.where(bgr_color[1]+err > g, 1, 0) & np.where(bgr_color[1]-err < g, 1, 0) &\
        np.where(bgr_color[2]+err > r, 1, 0) & np.where(bgr_color[2]-err < r, 1, 0)

male = color_mask(img, male_color) | color_mask(img, male_gray)
male = np.where(male==1, 255, 0)  

female = color_mask(img, female_color) | color_mask(img, female_gray)
female = np.where(female==1, 255, 0)  

cv.imshow("male", male.astype(np.uint8))
cv.imshow("female", female.astype(np.uint8))
cv.waitKey(0)
