﻿# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
import random

def mosaic_image(img, rect, size=(5,5), mtype=1):
	new_result = img.copy()

	for i in range(rect[0], rect[2], size[0]):
		for j in range(rect[1], rect[3], size[1]):
			dst = img[i : i + size[0], j : j + size[1]]

			if mtype == 1:
				dst = np.mean(dst, 1)
				dst = np.mean(dst, 0)

			elif mtype == 2:
				dst = np.max(dst, 1)
				dst = np.max(dst, 0)

			elif mtype == 3:
				dst = np.min(dst, 1)
				dst = np.min(dst, 0)

			new_result[i : i + size[0], j : j + size[1]] = dst

	return new_result

if __name__ == '__main__' :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True, \
			help = "Path to the input image")
	ap.add_argument("-s", "--start_point", type = int, \
 			nargs='+', default=[0, 0], \
			help = "Start point of the rectangle")
	ap.add_argument("-e", "--end_point", type = int, \
	 		nargs='+', default=[150, 100], \
			help = "End point of the rectangle")
	ap.add_argument("-z", "--size", type = int, \
	 		nargs='+', default=[15, 15], \
			help = "Mosaic Size")
	ap.add_argument("-t", "--type", type = int, \
	 		default=1, \
			help = "Mosaic Type")
	args = vars(ap.parse_args())

	filename = args["image"]
	sp = args["start_point"]
	ep = args["end_point"]
	size = args["size"]
	mtype = args["type"]

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename, cv2.IMREAD_COLOR)
	if(image is None):
		raise IOError("Cannot open the image")

	(rows, cols, _) = image.shape
	if(sp[0] < 0 or sp[1] < 0 or ep[0] > rows or ep[1] > cols):
		raise ValueError('Invalid Size')

	# list 연결
	rect = sp + ep

	# 모자이크 영상 생성
	result = mosaic_image(image, rect, size, mtype)

	# 영상 출력을 윈도우 생성
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	# 윈도우에 영상 출력
	cv2.imshow('image', result)

	# 사용자 입력 대기
	cv2.waitKey(0)
	# 윈도우 파괴
	cv2.destroyAllWindows()