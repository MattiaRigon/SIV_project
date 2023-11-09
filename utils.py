import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from show_image import show_image
import copy

lower_blue = np.array([50, 50, 50])  # Valori minimi per Hue, Saturation e Value
upper_blue = np.array([150, 255, 255])  # Valori massimi per Hue, Saturation e Value

lower_red = np.array([0, 100, 100])  # Valori minimi per Hue, Saturation e Value
upper_red = np.array([10, 255, 255])  # Valori massimi per Hue, Saturation e Value

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 25, 255])


lower_1 = lower_white
upper_1 = upper_white

lower_2 = lower_red
upper_2 = upper_red


font = cv2.FONT_HERSHEY_SIMPLEX


def find_line_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    if x2 - x1 == 0:
        return x2, None
    if y2 - y1 == 0:
        return None, y2
    m = (y2 - y1) / (x2 - x1)
    q = y1 - m * x1

    return m, q


def find_lines(src, image):

    clustered_lines = []
    lines = []
    M = []
    Q = []
    orizzontal = []
    vertical = []
    dst = cv2.Canny(src, 50, 150, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    # lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

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
    #         cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 10, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
            P1 = (l[0], l[1])
            P2 = (l[2], l[3])
            m, q = find_line_equation(P1, P2)
            if m is not None and q is not None:
                lines.append((m, q))
                M.append(m)
                Q.append(q)
            elif m is None:
                vertical.append(q)
            elif q is None:
                orizzontal.append(m)

    # plt.plot(M, Q, "o", color="red")

    # plt.savefig("linee.png")
    # plt.clf()

    if len(M) > 0:

        dbscan = DBSCAN(eps=100, min_samples=5)
        labels = dbscan.fit(np.array([M, Q]).T).labels_
        colors = np.array(["red", "green", "blue", "gray"])[labels]

        # plt.scatter(M, Q, c=colors, s=10)

        # plt.savefig("linee_cluster.png")

        # plt.clf()


def trovaCampoDaGioco(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    image_green = cv2.bitwise_and(image, image, mask=mask_green)
    gray_image = cv2.cvtColor(image_green, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cleaned_image = np.ones_like(image, dtype=np.uint8) * 255
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:
            cv2.drawContours(cleaned_image, [contour], -1, (0), thickness=cv2.FILLED)
    cleaned_image = 255 - cleaned_image
    cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
    result = copy.deepcopy(thresholded_image)
    result[cleaned_image == 255] = 255
    difference = result - thresholded_image
    contours, _ = cv2.findContours(
        difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 600:
            cv2.drawContours(result, [contour], -1, (0), thickness=cv2.FILLED)
    return result


def filter_and_find_players(player_image, binary_player_image):
    contours, hierarchy = cv2.findContours(
        binary_player_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    squad1 = np.zeros_like(binary_player_image)
    squad2 = np.zeros_like(binary_player_image)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:
            mask = np.zeros_like(binary_player_image)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            player_image = cv2.bitwise_and(player_image, player_image, mask=mask)
            player_hsv = cv2.cvtColor(player_image, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(player_hsv, lower_1, upper_1)
            res1_img = cv2.bitwise_and(player_image, player_image, mask=mask1)
            res1 = cv2.cvtColor(res1_img, cv2.COLOR_HSV2BGR)
            res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
            nzCount1 = cv2.countNonZero(res1)
            if nzCount1 > 20:
                _, thresholded_squad1 = cv2.threshold(res1, 1, 255, cv2.THRESH_BINARY)
                squad1 = cv2.add(squad1, thresholded_squad1)
            mask2 = cv2.inRange(player_hsv, lower_2, upper_2)
            res2_img = cv2.bitwise_and(player_image, player_image, mask=mask2)
            res2 = cv2.cvtColor(res2_img, cv2.COLOR_HSV2BGR)
            res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
            nzCount2 = cv2.countNonZero(res2)
            if nzCount2 > 20:
                _, thresholded_squad2 = cv2.threshold(res2, 1, 255, cv2.THRESH_BINARY)
                squad2 = cv2.add(squad2, thresholded_squad2)
    return squad1, squad2


def bounding_box_player(image, binary, squad):
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if squad == 1:
        color = (255, 0, 0)
        lower_color = lower_1
        upper_color = upper_1
    else:
        color = (0, 0, 255)
        lower_color = lower_2
        upper_color = upper_2
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 15 and h >= 15:
            rect = image[y : y + h, x : x + w]
            rect_hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(rect_hsv, lower_color, upper_color)
            res1_img = cv2.bitwise_and(rect_hsv, rect_hsv, mask=mask1)
            res1 = cv2.cvtColor(res1_img, cv2.COLOR_HSV2BGR)
            res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
            nzCount1 = cv2.countNonZero(res1)
            if nzCount1 > 50:
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            else:
                show_image(image[y : y + h, x : x + w])


def findPlayers(campo, image):
    contours, hierarchy = cv2.findContours(
        campo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    squad1 = np.zeros_like(campo)
    squad2 = np.zeros_like(campo)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        player_img = image[y : y + h, x : x + w]
        binary_player_img = campo[y : y + h, x : x + w]
        mask = np.zeros_like(binary_player_img)
        cv2.drawContours(mask, [c], -1, (255, 255, 255), thickness=cv2.FILLED)
        s1, s2 = filter_and_find_players(player_img, binary_player_img)
        squad1[y : y + h, x : x + w] = cv2.add(squad1[y : y + h, x : x + w], s1)
        squad2[y : y + h, x : x + w] = cv2.add(squad2[y : y + h, x : x + w], s2)

    # s1 = cv2.bitwise_and(image, image,mask=squad1)
    # show_image(s1)
    # s2 = cv2.bitwise_and(image, image,mask=squad2)
    # show_image(s2)

    bounding_box_player(image, squad1, 1)
    bounding_box_player(image, squad2, 0)

    show_image(image)
