# 필요한 패키지를 import함
from __future__ import print_function
from random import seed, randint
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def findContours(bin):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    img = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    (contours, hierarchy) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return img, contours

	# 	# 무작위 색으로 모든 연결요소의 외곽선 그림
	# seed(9001)
	# contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	# for (i, c) in enumerate(contours):
	# 	r = randint(0, 256)
	# 	g = randint(0, 256)
	# 	b = randint(0, 256)
	# 	cv2.drawContours(contour_img, [c], 0, (b,g,r), 2)

if __name__ == "__main__" :
	# 명령행 인자 처리
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, \
                    help = 'Path to the input image')
    args = vars(ap.parse_args())

    filename = args['image']

	# OpenCV를 사용하여 영상 데이터 로딩
    image = cv2.imread(filename)

	# Grayscale 영상으로 변환한 후
	# 가우시안 평활화 및 임계화 수행
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    bininary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 2)

	# 연결요소 생성
    contour_img, contours = findContours(bininary)

	# 0번(1st) 연결요소의 외곽선만을 포함하는 영상 생성
    new_img = np.zeros_like(contour_img, dtype="uint8")
    for idx in range (len(contours)):
        cntr = sorted(contours, key=cv2.contourArea, reverse=True)[idx]
        cv2.drawContours(new_img, [cntr], 0, (255,255,255), -1)

    plt.subplot(1,3,1), plt.imshow(gray, cmap='gray')
    plt.title('grayscale and blur'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2), plt.imshow(bininary, cmap='gray')
    plt.title('threshold'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,3), plt.imshow(new_img, cmap='gray')
    plt.title('contour'), plt.xticks([]), plt.yticks([])
    plt.show()