from __future__ import print_function
import argparse
import PPM.PPM_P6 as ppm
import array
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, \
                help="Path to the input image")

ap.add_argument("-o", "--output", required=True, \
                help="Path to the output image")

ap.add_argument("-l", "--location", type=int, \
                default=[0, 0], nargs="+", \
                help="Location of the output image")

ap.add_argument("-s", "--size", type=int, \
                default=[50, 30], nargs="+", \
                help="Size of the output image")

ap.add_argument("-c", "--color", type=int, \
                default=[255, 0, 0], nargs="+", \
                help="Color of the output image")

args = vars(ap.parse_args())

infile = args["input"]
outfile = args["output"]

rec_location = args['location']

rec_size = args["size"]
rec_height = rec_size[0]
rec_width = rec_size[1]

rec_color = args["color"]

ppm_p6 = ppm.PPM_P6()

(width, height, maxval, bitmap) = ppm_p6.read(infile)

rectangle = array.array('B', bitmap)
rectangle = np.array(rectangle)
rectangle = rectangle.reshape((height,width,3))

rectangle[rec_location[0] : rec_location[0] + rec_width, rec_location[1] : rec_location[1] + rec_height] = [rec_color[0], rec_color[1], rec_color[2]]

rectangle = rectangle.reshape(height*width*3)
rectangle = bytes(rectangle)

ppm_p6.write(width, height, maxval, rectangle, outfile)