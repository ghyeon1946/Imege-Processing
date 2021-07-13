# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
from matplotlib import pyplot as plt

def histogram(img):

	# 결과 히스토그램을 저장할 리스트
	hist = []

	# 코드 작성
	length = len(img.shape)

	# 그레이스케일
	if length == 2:
		temp = cv2.calcHist([img], [0], None, [256], [0, 256])
		hist.append(temp)

	# 칼라
	elif length == 3:
		for i in range(0, 3):
			element = cv2.calcHist([img], [i], None, [256], [0, 256])
			hist.append(element)

	return hist

def scaleHistogram(img, rge):
    
    # 결과 히스토그램을 저장할 리스트
    hist = []

	# 코드 작성
    length = len(img.shape)

    if length == 2:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                index = img.item(i, j)
                img[i, j] = 2.8 * index

                if(img[i,j] > int(rge[1])):
                    img[i,j] = int(rge[1])

        histb = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist.append(histb)

    elif length == 3:
        for i in range(3):
            for j in range(img.shape[0]):
                for k in range(img.shape[1]):
                    image[j, k, i] = 2.8 * image[j, k, i]

                    if (img[j, k, i] > int(rge[1])):
                        img[j, k, i] = int(rge[1])

            histb = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist.append(histb)

    return hist

def slideHistogram(img, rge, slide):
    # 결과 히스토그램을 저장할 리스트
    hist = []

	# 코드 작성
    length = len(img.shape)

    if length == 2:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                index = img.item(i, j)
                img[i, j] = index + int(slide[0])

                if (img[i, j] > int(rge[1])):
                    img[i, j] = int(rge[1])

        histb = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist.append(histb)

    elif length == 3:
        for i in range(3):
            for j in range(img.shape[0]):
                for k in range(img.shape[1]):
                    image[j, k, i] = image[j, k, i] + int(slide[0])

                    if (img[j, k, i] > int(rge[1])):
                        img[j, k, i] = int(rge[1])

            histb = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist.append(histb)

    return hist

if __name__ == '__main__' :
	# 명령행 인자 처리
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True, \
            help = 'Path to the input image')
    ap.add_argument('-r', '--range', required = True, \
            nargs='+', default=[0, 250], \
            help = 'range')
    ap.add_argument('-s', '--slide', required = True, \
            nargs='+', default=50, \
            help = 'range')
    args = vars(ap.parse_args())

    filename = args['image']
    rge = args['range']
    slide = args['slide']

	# OpenCV를 사용하여 영상 데이터 로딩
    image = cv2.imread(filename)

	# 히스토그램 계산
    hist = histogram(image)

	# 히스토그램 출력
    if len(hist) == 1: 
        plt.subplot(3, 2, 1), plt.imshow(image, 'gray')
        plt.title('image'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 2, 2), plt.plot(hist[0])
        plt.title('histogram'), plt.xlim([0,256])

    else:
        color = ('b', 'g', 'r')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(3, 2, 1), plt.imshow(image)
        plt.title('image'), plt.xticks([]), plt.yticks([])

        for n, col in enumerate(color):
            plt.subplot(3, 2, 2)
            plt.plot(hist[n], color = col)

        plt.title('histogram'), plt.xlim([0,256])

    hist2 = scaleHistogram(image, rge)

    if len(hist) == 1:
        plt.subplot(3,2,3), plt.imshow(image, 'gray')
        plt.title('scale'), plt.xticks([]), plt.yticks([])
        plt.subplot(3,2,4), \
        plt.plot(hist2[0], color='k')
        plt.title('histogram'), plt.xlim([0,256])

    else:
        color = ('r', 'g', 'b')

        plt.subplot(3, 2, 3), plt.imshow(image)
        plt.title('scale'), plt.xticks([]), plt.yticks([])

        for n, col in enumerate(color):
            plt.subplot(3, 2, 4)
            plt.plot(hist2[n], color=col)

        plt.title('histogram'), plt.xlim([0, 256])

    hist3 = slideHistogram(image, rge, slide)

    if len(hist) == 1:
        plt.subplot(3, 2, 5), plt.imshow(image, 'gray')
        plt.title('slide'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 2, 6), \
        plt.plot(hist3[0], color='k')
        plt.title('histogram'), plt.xlim([0, 256])

    else:
        color = ('r', 'g', 'b')

        plt.subplot(3, 2, 5), plt.imshow(image)
        plt.title('slide'), plt.xticks([]), plt.yticks([])

        for n, col in enumerate(color):
            plt.subplot(3, 2, 6)
            plt.plot(hist3[n], color=col)
            
        plt.title('histogram'), plt.xlim([0, 256])

    plt.show()