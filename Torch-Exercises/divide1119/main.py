import cv2
import numpy as np
import matplotlib.pyplot as plt

prob_list = [[0 for i in range(0, 256)] for j in range(0, 256)]
prob_list = np.array(prob_list)
img = cv2.imread("lena.png", 0)
hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])
img = np.array(img)

img_out = []

for items in img:
    img_line = []
    for i in range(0, items.size, 2):
        filter_low = 0.5 * items[i] + 0.5 * items[i + 1]
        img_line.append(filter_low)
    for i in range(0, items.size, 2):
        filter_high = 0.5 * items[i] - 0.5 * items[i + 1]
        img_line.append(filter_high)
    img_out.append(img_line)

img_out = np.array(img_out)

img_out2 = [[None for i in range(0, 512)] for j in range(0, 512)]

for j in range(0, 512):
    for i in range(0, 512, 2):
        img_out2[(i >> 1)][j ] = img_out[i][j] * 0.5 + img_out[i+1][j] * 0.5
        img_out2[256 + (i >> 1)][j ] = img_out[i][j] * 0.5 - img_out[i+1][j] * 0.5

img_out = np.array(img_out2)
cv2.imshow("img",img_out/100)

cv2.waitKey()
