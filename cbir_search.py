# 필요한 패키지 임포트
import numpy as np
import argparse
import pickle
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt

def histogram1D(img):
	# 1차원 히스토그램 계산
	hist = cv2.calcHist([img], [0], None, [256], [0, 256])
	hist = cv2.normalize(hist, hist)

	# 결과 히스토그램 반환
	return hist

def histogram2D(img):
	# 2차원 히스토그램 계산
	hist = cv2.calcHist( [img], [0, 1], None, \
				[180, 256], [0, 180, 0, 256] )
	hist = cv2.normalize(hist, hist)

	# 결과 히스토그램 반환
	return hist

def histogram3D(img):
	# 2차원 히스토그램 계산
	hist = cv2.calcHist( [img], [0, 1, 2], None, \
				[32, 32, 32], [0, 256, 0, 256, 0, 256] )
	hist = cv2.normalize(hist, hist)

	# 결과 히스토그램 반환
	return hist

def calc_similarity(index):
	similarity = {} 
	
	### code 작성

	for key, values in index.items():
		similarity[key] = cv2.compareHist(index[key], histq, cv2.HISTCMP_BHATTACHARYYA)
		
	similarity = OrderedDict(sorted(similarity.items(), key=lambda x: x[1], reverse= False))

	return similarity

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required = True,
		help = "Path to the directory that contains the images we just indexed")
	ap.add_argument("-i", "--index", required = True,
		help = "Path to where we stored our index")
	ap.add_argument("-q", "--query", required = True,
		help = "Path to query image")
	args = vars(ap.parse_args())

	# 질의 영상 불러오고 화면 출력
	queryImage = cv2.imread(args["query"])

	# 질의 영상에 대한 히스토그램 생성
	histq = histogram2D(queryImage)

	# 인덱스 파일 불러오기
	index = pickle.loads(open(args["index"], "rb").read())

	# 유사도 계산
	similarity = calc_similarity(index)
	print(similarity)

	fig = plt.figure()
	axes = []
	rows, cols = 2, 3
	for n, item in enumerate(similarity):
		print(n+1, item)
		img = cv2.imread(item)
		axes.append(fig.add_subplot(rows, cols, n+1))
		subplot_title = item
		axes[-1].set_title(subplot_title)
		plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

		if n >= 5:
			break
	plt.show()