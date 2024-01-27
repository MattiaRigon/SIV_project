import cv2
from utils import *

def preprocess(image):
        
        new_dimension = (2048, 1080)
        image = cv2.resize(image, new_dimension)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Definisci i range di colore per il verde (puoi adattarli per ottenere la tonalità desiderata)
        lower_green = np.array([25, 40, 40])  # Soglia inferiore per il verde in HSV
        upper_green = np.array([100, 255, 255])  # Soglia superiore per il verde in HSV

        # Crea una maschera per isolare la parte verde dell'immagine
        mask = cv2.inRange(hsv, lower_green, upper_green)


        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

        for c in contours:
            area = cv2.contourArea(c)
            if area > 10000:
                cv2.drawContours(image, c, -1, (0, 0, 255), 10)
        # Applica la maschera all'immagine originale per ottenere solo la parte verde
        green_part = cv2.bitwise_and(image, image, mask=mask)

        return image


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


def define_region(vidcap):
    success, image = vidcap.read()


    green_image = preprocess(image)
    # find_lines(eroded_image, image)
    
    # show_image(eroded_image)
    show_image(green_image)






if __name__ == "__main__":
    # video_input/cagliari-chievo/2h-left-5min.avi
    # video_input/chievo-juve/1h-left.avi
    vidcap = cv2.VideoCapture("video_input/chievo-juve/1h-left.avi")
    define_region(vidcap)


