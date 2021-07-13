from __future__ import print_function
import argparse
import numpy as np
import cv2
import os
import glob
import timeit

def detectBarcode(img, imagePath):
    points = [0, 0, 0, 0]

    x_sobel = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=-1)
    y_sobel = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=-1)

    sobel_x = np.uint8(np.absolute(x_sobel))
    loc1 = imagePath.rfind("\\")
    loc2 = imagePath.rfind(".")
    fname = 'sobel/' + imagePath[loc1 + 1: loc2] + '_sobel_x.jpg'
    cv2.imwrite(fname, sobel_x)

    sobel_y = np.uint8(np.absolute(y_sobel))
    loc1 = imagePath.rfind("\\")
    loc2 = imagePath.rfind(".")
    fname = 'sobel/' + imagePath[loc1 + 1: loc2] + '_sobel_y.jpg'
    cv2.imwrite(fname, sobel_y)

    x_dst = cv2.subtract(x_sobel, y_sobel)
    y_dst = cv2.subtract(y_sobel, x_sobel)

    loc1 = imagePath.rfind("\\")
    loc2 = imagePath.rfind(".")
    fname = 'sub/' + imagePath[loc1 + 1: loc2] + '_x-y.jpg'
    cv2.imwrite(fname, x_dst)

    loc1 = imagePath.rfind("\\")
    loc2 = imagePath.rfind(".")
    fname = 'sub/' + imagePath[loc1 + 1: loc2] + '_y-x.jpg'
    cv2.imwrite(fname, y_dst)

    x_dst[x_dst < 0] = 0
    y_dst[y_dst < 0] = 0

    x_dst = cv2.GaussianBlur(x_dst, (15, 15), 0)
    x_th, x_dst = cv2.threshold(x_dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (92, 1))
    x_dst = cv2.morphologyEx(x_dst, cv2.MORPH_CLOSE, x_kernel)

    y_dst = cv2.GaussianBlur(y_dst, (9, 9), 0)
    y_th, y_dst = cv2.threshold(y_dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    y_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 92))
    y_dst = cv2.morphologyEx(y_dst, cv2.MORPH_CLOSE, y_kernel)

    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

    x_dst = cv2.erode(x_dst, kernal, iterations = 12)
    y_dst = cv2.erode(y_dst, kernal, iterations = 12)

    x_dst = cv2.dilate(x_dst, kernal, iterations = 12)
    y_dst = cv2.dilate(y_dst, kernal, iterations = 12)

    loc1 = imagePath.rfind("\\")
    loc2 = imagePath.rfind(".")
    fname = 'post/' + imagePath[loc1 + 1: loc2] + '_post_x.jpg'
    cv2.imwrite(fname, x_dst)

    loc1 = imagePath.rfind("\\")
    loc2 = imagePath.rfind(".")
    fname = 'post/' + imagePath[loc1 + 1: loc2] + '_post_y.jpg'
    cv2.imwrite(fname, y_dst)

    (contours_x, hierarchy) = cv2.findContours(x_dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (contours_y, hierarchy) = cv2.findContours(y_dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_x) == 0 and len(contours_y) == 0:
        return [0, 0, 0, 0]

    if len(contours_x) == 0:
        x_cntr = sorted(contours_y, key=cv2.contourArea, reverse=True)[0]
        x1, y1, w1, h1 = cv2.boundingRect(x_cntr)
        return [x1, y1, x1 + w1, y1 + h1]

    if len(contours_y) == 0:
        y_cntr = sorted(contours_x, key=cv2.contourArea, reverse=True)[0]
        x2, y2, w2, h2 = cv2.boundingRect(y_cntr)
        return [x2, y2, x2 + w2, y2 + h2]

    contours_x = sorted(contours_x, key=cv2.contourArea, reverse=True)[0]
    contours_y = sorted(contours_y, key=cv2.contourArea, reverse=True)[0]

    x, y, w, h = cv2.boundingRect(contours_x)
    x2, y2, w2, h2 = cv2.boundingRect(contours_y)

    if w * h > w2 * h2:
        return [x, y, x+w, y+h]

    else:
        return [x2, y2, x2 + w2, y2 + h2]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required = True, help = "path to the dataset folder")
    ap.add_argument("-r", "--detectset", required = True, help = "path to the detectset folder")
    ap.add_argument("-f", "--detect", required = True, help = "path to the detect file")
    args = vars(ap.parse_args())
    
    dataset = args["dataset"]
    detectset = args["detectset"]
    detectfile = args["detect"]

    # 결과 영상 저장 폴더 존재 여부 확인
    if(not os.path.isdir(detectset)):
        os.mkdir(detectset)

    # 결과 영상 표시 여부
    verbose = False

    # 검출 결과 위치 저장을 위한 파일 생성
    f = open(detectfile, "wt", encoding="UTF-8")  # UT-8로 인코딩
    
    start = timeit.default_timer()

    # 바코드 영상에 대한 바코드 영역 검출
    for imagePath in glob.glob(dataset + "/*.jpg"):
        print(imagePath, '처리중...')

        # 영상을 불러오고 그레이 스케일 영상으로 변환
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 바코드 검출
        points = detectBarcode(gray, imagePath)

        # 바코드 영역 표시
        detectimg = cv2.rectangle(image, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 2)  # 이미지에 사각형 그리기

        # 결과 영상 저장
        loc1 = imagePath.rfind("\\")
        loc2 = imagePath.rfind(".")
        fname = 'result/' + imagePath[loc1 + 1: loc2] + '_res.jpg'
        cv2.imwrite(fname, detectimg)

        # 검출한 결과 위치 저장
        f.write(imagePath[loc1 + 1: loc2])
        f.write("\t")
        f.write(str(points[0]))
        f.write("\t")
        f.write(str(points[1]))
        f.write("\t")
        f.write(str(points[2]))
        f.write("\t")
        f.write(str(points[3]))
        f.write("\n")

        if verbose:
            cv2.imshow("image", image)
            cv2.waitKey(0)

    end = timeit.default_timer()

    print("모든 영상을 처리하는데 소요된 시간 : ", end - start)