import cv2 as cv
import numpy as np

img = cv.imread('BeforeSleep448_2030_screenshot.png')

# 從圖中對顏色取樣
male_color = img[800, 1300]
male_gray = img[700, 1300]

female_color = img[800, 1500]
female_gray = img[700, 1500]

# 產生遮罩
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

male_mask = color_mask(img, male_color) | color_mask(img, male_gray)
male_mask = np.where(male_mask==1, 255, 0).astype(np.uint8)

female_mask = color_mask(img, female_color) | color_mask(img, female_gray)
female_mask = np.where(female_mask==1, 255, 0).astype(np.uint8)

# 去除小面積獨立塊
k_size  = np.array([37, 37])
op = cv.MORPH_HITMISS
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, k_size)
male_mask   = cv.morphologyEx(  male_mask, op, kernel)
female_mask = cv.morphologyEx(female_mask, op, kernel)

# 把少掉的邊緣長回來
k_size  = np.array([5, 5])
op = cv.MORPH_DILATE
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, k_size)
for i in range(10):
    male_mask   = cv.morphologyEx(  male_mask, op, kernel)
    female_mask = cv.morphologyEx(female_mask, op, kernel)

# 計算面積
male_area   = np.count_nonzero(male_mask)
female_area = np.count_nonzero(female_mask)

# 產生示意圖
male   = cv.bitwise_and(img, img, mask=male_mask)
female = cv.bitwise_and(img, img, mask=female_mask)
blend_image = cv.addWeighted(male+female, 0.8, img, 0.2, 1)

cv.imshow("blend_image, area= {} | {}".format(male_area, female_area), blend_image)
cv.waitKey(1)

# =====================================
# 等比例縮放，把面積變成1:1
# =====================================

ratio = female_area/male_area

# 找到水平縮放起終點
start, end = 0, 0

female_mask_project_to_x = np.sum(female_mask, axis=0)
for i, v in enumerate(female_mask_project_to_x):
    if v != 0:
        start = i
        break

for i, v in enumerate(female_mask_project_to_x[::-1]):
    if v != 0:
        end = i
        break

female_mask[:,start] = 255
female_mask[:,-end] = 255
cv.imshow("female_mask", female_mask)
cv.imshow("female_mask cut", female_mask[:,start:-end])
cv.waitKey(0)


