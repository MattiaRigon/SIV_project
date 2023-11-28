import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from show_image import show_image
import copy
import cv2
from mpl_toolkits.mplot3d import Axes3D
from create_classification_model import *
from classification import predict 

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

def get_hsv_range_from_image(img):

    # Converti l'immagine da BGR a HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calcola l'istogramma per H, S, V
    hist_h = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])

    # Calcola i percentili per H, S, V che includono il 90% delle occorrenze
    h_min = np.percentile(np.nonzero(hist_h)[0], 10)  # Percentile 5
    h_max = np.percentile(np.nonzero(hist_h)[0], 80)  # Percentile 95
    s_min = np.percentile(np.nonzero(hist_s)[0], 10)  # Percentile 5
    s_max = np.percentile(np.nonzero(hist_s)[0], 80)  # Percentile 95
    v_min = np.percentile(np.nonzero(hist_v)[0], 10)  # Percentile 5
    v_max = np.percentile(np.nonzero(hist_v)[0], 80)  # Percentile 95


    return {
        "Hue Range": (h_min, h_max),
        "Saturation Range": (s_min, s_max),
        "Value Range": (v_min, v_max)
    }

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
    #         P1 = (l[0], l[1])
    #         P2 = (l[2], l[3])
    #         m, q = find_line_equation(P1, P2)
    #         if m is not None and q is not None:
    #             lines.append((m, q))
    #             M.append(m)
    #             Q.append(q)
    #         elif m is None:
    #             vertical.append(q)
    #         elif q is None:
    #             orizzontal.append(m)

    # # plt.plot(M, Q, "o", color="red")

    # # plt.savefig("linee.png")
    # # plt.clf()

    # if len(M) > 0:

    #     dbscan = DBSCAN(eps=100, min_samples=5)
    #     labels = dbscan.fit(np.array([M, Q]).T).labels_
    #     # colors = np.array(["red", "green", "blue", "gray"])[labels]

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
    return thresholded_image
    # contours, _ = cv2.findContours(
    #     thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    # )
    # cleaned_image = np.ones_like(image, dtype=np.uint8) * 255
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area > 10:
    #         cv2.drawContours(cleaned_image, [contour], -1, (0), thickness=cv2.FILLED)
    # cleaned_image = 255 - cleaned_image
    # cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
    # result = copy.deepcopy(thresholded_image)
    # result[cleaned_image == 255] = 255
    # difference = result - thresholded_image
    # contours, _ = cv2.findContours(
    #     difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    # )
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area > 600:
    #         cv2.drawContours(result, [contour], -1, (0), thickness=cv2.FILLED)
    # return result


def filter_and_find_players(player_image, binary_player_image):
    contours, hierarchy = cv2.findContours(
        binary_player_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    squad1 = np.zeros_like(binary_player_image)
    squad2 = np.zeros_like(binary_player_image)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            # show_image(player_image)
            mask = np.zeros_like(binary_player_image)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            player_image = cv2.bitwise_and(player_image, player_image, mask=mask)
            player_hsv = cv2.cvtColor(player_image, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(player_hsv, lower_1, upper_1)
            res1_img = cv2.bitwise_and(player_image, player_image, mask=mask1)
            res1 = cv2.cvtColor(res1_img, cv2.COLOR_HSV2BGR)
            res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
            nzCount1 = cv2.countNonZero(res1)
            if nzCount1 > -1:
                _, thresholded_squad1 = cv2.threshold(res1, 1, 255, cv2.THRESH_BINARY)
                squad1 = cv2.add(squad1, thresholded_squad1)
            mask2 = cv2.inRange(player_hsv, lower_2, upper_2)
            res2_img = cv2.bitwise_and(player_image, player_image, mask=mask2)
            res2 = cv2.cvtColor(res2_img, cv2.COLOR_HSV2BGR)
            res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
            nzCount2 = cv2.countNonZero(res2)
            if nzCount2 > -1:
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
        # if w > 15 and h >= 15:
        rect = image[y : y + h, x : x + w]
        rect_hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(rect_hsv, lower_color, upper_color)
        res1_img = cv2.bitwise_and(rect_hsv, rect_hsv, mask=mask1)
        res1 = cv2.cvtColor(res1_img, cv2.COLOR_HSV2BGR)
        res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
        nzCount1 = cv2.countNonZero(res1)
        if nzCount1 > 10:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            # show_image(image[y : y + h, x : x + w])

def calculate_hist(image):
    hist_hue = cv2.calcHist([image], [0], None, [180], [0, 180])
    hist_saturation = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([image], [2], None, [256], [0, 256])
    
    hist_hue = hist_hue[1:]  # Remove the first element from hist_hue
    hist_saturation = hist_saturation[1:-1]  # Remove the first and last element from hist_saturation
    hist_value = hist_value[1:-1]  # Remove the first and last element from hist_value



    # plt.plot( ist_hue color='r')
    plt.title('Histogram - Hue')
    plt.xlabel('Hue')
    plt.ylabel('Frequency')

    plt.plot(hist_saturation, color='g')
    plt.title('Histogram - Saturation')
    plt.xlabel('Saturation')
    plt.ylabel('Frequency')

    plt.plot(hist_value, color='b')
    plt.title('Histogram - Value')
    plt.xlabel('Value')
    plt.ylabel('Frequency')


    plt.savefig("histogram_player_img.png")
    plt.clf()


    # linee = find_lines(campo, image)
    # show_image(campo)
    # # show_image(linee)

def plot_hsv_space(image):



    # Estrai i canali H, S e V
    h, s, v = cv2.split(image)

    # Crea una figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotta i punti nello spazio 3D
    ax.scatter(h.flatten(), s.flatten(), v.flatten(), c=image.reshape(-1, 3) / 255.0, marker='o')

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(image)

    dbscan = DBSCAN(eps=100, min_samples=5)
    labels = dbscan.fit(image).labels_#dbscan.fit(np.array([h, s,v]).T).labels_
    colors = np.array(["red", "green", "blue", "gray"])[labels]
    # Etichette degli assi
    ax.set_xlabel('H')
    ax.set_ylabel('S')
    ax.set_zlabel('V')

    # Salva l'immagine
    plt.savefig('grafico_3d_colore_HSV.png')  # Modifica il nome del file come preferisci

    # Chiudi la finestra di visualizzazione
    plt.close()


def findPlayers(campo, image,svm_classifier):



    contours, hierarchy = cv2.findContours(
        campo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    squad1 = np.zeros_like(campo)
    squad2 = np.zeros_like(campo)
    all =  np.zeros_like(campo)
    # show_image(player_img)
    # show_image(campo)
    # show_image(image)
    image_masked = cv2.bitwise_and(image, image, mask=campo)
    # show_image(image_masked)
    # show_image(campo)

    # campo_canny = cv2.Canny(image_masked, 300, 600, None, 3)

    # show_image(campo_canny)
    # calculate_hist(image_masked)
    # plot_hsv_space(image_masked)
    tmp = image.copy()
    i=0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # if w > 15 and h >= 15:
        if area > 20:
            # show_image(image[y : y + h, x : x + w])
            player_img = image[y : y + h, x : x + w]
            binary_player_img = campo[y : y + h, x : x + w]
            masked_image = cv2.bitwise_and(player_img, player_img, mask=binary_player_img)
            # show_image(masked_image)
            # cv2.imwrite(f"test/masked_image{i}.png",masked_image)
            label = predict(svm_classifier,masked_image)

            if label == [0]:
                cv2.rectangle(image, (x, y), (x + w, y + h),  (255, 0, 0), 2)
            else :
                cv2.rectangle(image, (x, y), (x + w, y + h),  (0, 0, 255), 2)

            # mask = np.zeros_like(binary_player_img)
            # cv2.drawContours(tmp, [c], -1, (255, 255, 255), thickness=cv2.FILLED)
            # cv2.rectangle(tmp, (x, y), (x + w, y + h),  (0, 255, 0), 3)
            # show_image(tmp)
            # show_image(player_img)
            # show_image(binary_player_img)
            # print(model.predict(filter_image_green(player_img)))
            # s1, s2 = filter_and_find_players(player_img, binary_player_img)
            # squad1[y : y + h, x : x + w] = cv2.add(squad1[y : y + h, x : x + w], s1)
            # squad2[y : y + h, x : x + w] = cv2.add(squad2[y : y + h, x : x + w], s2)
            # all[y : y + h, x : x + w] = cv2.add(all[y : y + h, x : x + w], binary_player_img)
            # i=i+1

    # s1 = cv2.bitwise_and(image, image,mask=squad1)
    # show_image(s1)
    # s2 = cv2.bitwise_and(image, image,mask=squad2)
    # show_image(s2)


    # show_image(all)
    # bounding_box_player(image, squad1, 1)
    # bounding_box_player(image, squad2, 0)

    show_image(image)
