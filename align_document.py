from __future__ import print_function
import argparse
import numpy as np
import cv2

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True,
                    help='path to the input image')
    args = vars(ap.parse_args())

    image = args['image']

    image = cv2.imread(image)

    rows, cols, _ = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, 0, 255)

    (contours, _) = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(5):
        cntr = sorted(contours, key=cv2.contourArea, reverse=True)[i]
        epsilon = 0.01 * cv2.arcLength(cntr, True)
        approx = cv2.approxPolyDP(cntr, epsilon, True)

        nums = len(approx)
        if nums == 4:
            break

    pts1 = np.float32([approx[3][0], approx[0][0], approx[2][0], approx[1][0]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(gray, matrix, (cols, rows))

    threshold = cv2.adaptiveThreshold(perspective, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)

    cv2.imshow('image', threshold)
    cv2.waitKey(0)