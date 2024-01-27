"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import LineString



def simplify_line(points):
    line = LineString(points)
    tolerance = 1.0
    simplified_line = line.simplify(tolerance, preserve_topology=False)
    x, y = simplified_line.xy
    plt.plot(x, y)
    plt.savefig("line_optimized.png")


def compute_centroids(cartesian_points):

    # Extract x and y coordinates from the cartesian points
    x = [c[0] for c in cartesian_points]
    y = [c[1] for c in cartesian_points]

    # Initialize variables for sum of x and y coordinates
    sum_x, sum_y = 0, 0

    # Calculate the sum of x and y coordinates
    for i in x:
        sum_x += i
    for j in y:
        sum_y += j

    # Calculate the mean of x and y coordinates
    mean_x, mean_y = 1, 1  # Default values to prevent division by zero
    if len(x) != 0:
        mean_x = sum_x / len(x)
        mean_y = sum_y / len(y)

    # Return the polar coordinates
    return mean_x, mean_y

def find_lines(src, image):
    dst = cv.Canny(src, 50, 150, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    # lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 10, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(src, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
    #         x =[l[0], l[2]]
    #         y = [l[1], l[3]]
    #         p1 = (l[0], l[1])
    #         P2 = (l[2], l[3])
    #         plt.plot(
    #             x,
    #             y,
    #             label="Line Connecting Two Points",
    #             color="blue",
    #             linestyle="-",
    #             marker="o",
    #         )

    # # Customize the plot
    # plt.title("Line Connecting Two Points")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.grid(True)

    # plt.savefig("linee.png")

    return cdstP
