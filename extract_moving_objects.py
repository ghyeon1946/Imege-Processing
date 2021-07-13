# 필요한 패키지를 import함
from __future__ import print_function
from random import seed, randint
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def movingObjectsDetect(img):
	# 타원형의 구조적 요소 생성
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	(contours, hierarchy) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	seed(9001)

	contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	for (i, c) in enumerate(contours):
		r = randint(0, 256)
		g = randint(0, 256)
		b = randint(0, 256)
		cv2.drawContours(contour_img, [c], 0, (b,g,r), -1)

	return contour_img

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-v', '--video', required = False, \
			help = 'Path to the input video')
	args = vars(ap.parse_args())

	fvideo = args.get("video")

	if fvideo is None:
		camera = cv2.VideoCapture(0)
	else:
		camera = cv2.VideoCapture(args["video"])

	model = cv2.createBackgroundSubtractorMOG2()

	while True:
		# 현재 프레임 획득
		#  frame: 획득한 비디오 프레임
		#  retfal: 프레임 획득이 되지 못하면 False
		(retval, frame) = camera.read()
	 
		# 비디오 파일의 마지막 위치 도착 확인
		if fvideo is not None and not retval:
			break

		frame1 = model.apply(frame)
		frame1 = movingObjectsDetect(frame1)
		
		# 결과 영상 출력
		cv2.imshow("images", np.hstack((frame,frame1)))

		# 'q' 키를 누르면 종료
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	 
	# 비디오 카메라 정리
	camera.release()