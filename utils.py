import cv2
import numpy as np
from classification import predict 
import random
import matplotlib
import matplotlib.pyplot as plt

from player import Player 
matplotlib.use("TkAgg")


font = cv2.FONT_HERSHEY_SIMPLEX

counter = 0

def show_image(output_img,name = ""):
    """
    Display the output image in a window.

    Parameters:
    - output_img: numpy.ndarray
        The output image to be displayed.
    - name: str, optional
        The name of the window. Default is an empty string.

    Returns:
    None

    """

    cv2.imshow("Output Image", output_img.astype(np.uint8))
    while True:
        key = cv2.waitKey(1) & 0xFF

        # Break the loop if the 'q' key is pressed or the window is closed
        if key == ord('q') or cv2.getWindowProperty("Output Image", cv2.WND_PROP_VISIBLE) < 1:
            break

def generate_photo_dataset(nome_file,n,pts):
    
    """
    Generate a photo dataset from a video file.

    Args:
        nome_file (str): The name of the video file.
        n (int): The number of frames to sample from the video.
        pts (list): The list of points for preprocessing.

    Returns:
        numpy.ndarray: The fixed points image.

    Raises:
        None
    """

    vidcap = cv2.VideoCapture(f"video_input/{nome_file}")
    fixed_points = np.zeros((1080, 2048), dtype=np.int32)
    processed_images = []

    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_indices = random.sample(range(0, frame_count), n)
    frames = []
    for idx in random_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = vidcap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Error reading the frame {idx}")

    for frame in frames:
        eroded_image, image = preprocess(frame,pts)
        fixed_points = fixed_points + eroded_image

        processed_images.append([eroded_image,image])

    fixed_points = np.where(fixed_points > 255*(len(frames)-1)*0.7, 255, 0).astype(np.uint8)
    fixed_points_img_name = str(nome_file).replace(".avi","")+ "/fixed_points.png"
    
    cv2.imwrite(f"datasets{fixed_points_img_name}",fixed_points)
    for eroded_image, image in processed_images:
        eroded_image = eroded_image - fixed_points
        eroded_image = np.where(eroded_image == 255, 255, 0).astype(np.uint8)
        findPlayers(eroded_image, image,None,creating_dataset=True,directory_name=nome_file.replace('.avi',''))


    vidcap.release()

    return fixed_points


def preprocess(image,pts,fixed_points=[]):

    """
    Preprocesses an image by resizing it, creating a mask based on given points, 
    applying bitwise operations, converting to grayscale, and performing dilation and erosion.

    Args:
        image (numpy.ndarray): The input image.
        pts (list): List of points to create a polygon mask.
        fixed_points (list, optional): List of fixed points to subtract from the mask. Defaults to [].

    Returns:
        tuple: A tuple containing the eroded image and the original image.
    """
                
    new_dimension = (2048, 1080)
    image = cv2.resize(image, new_dimension)

    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = np.zeros((1080, 2048), dtype=np.uint8)  # Creazione di un'immagine vuota per la maschera
    cv2.fillPoly(mask, [pts], 255)  # Disegna il poligono sulla maschera

    _image = cv2.bitwise_and(image, image, mask=mask)
    mask = deleteBackground(_image)
    masked_image = cv2.bitwise_and(_image, _image, mask=mask)
    lower = np.array([100, 0, 0])
    upper = np.array([180, 255, 255])
    mask_hist = cv2.inRange(masked_image, lower, upper)
    inverse_mask_hist = cv2.bitwise_not(mask_hist)  # Create inverse mask
    inverse_image_hist = cv2.bitwise_and(masked_image, masked_image, mask=inverse_mask_hist)
    gray_image = cv2.cvtColor(inverse_image_hist, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    if len(fixed_points) != 0 :
        binary_image = binary_image - fixed_points
        binary_image = np.where(binary_image == 255, 255, 0).astype(np.uint8)

    dilated_image = cv2.dilate(binary_image, None, iterations=1)
    eroded_image = cv2.erode(dilated_image, None, iterations=1)

    return eroded_image, image

def deleteBackground(image):

    """
    Deletes the background of an image by converting it to HSV, creating a mask based on the green color,
    and applying bitwise operations.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The mask of the image.
    """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    image_green = cv2.bitwise_and(image, image, mask=mask_green)
    gray_image = cv2.cvtColor(image_green, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

def findPlayers(campo, image,svm_classifier,creating_dataset=False,directory_name=""):

    """
    Finds the soccer players in the image by detecting contours and creating a mask.

    Args:
        campo (numpy.ndarray): The input image.
        image (numpy.ndarray): The input image.
        svm_classifier (object): The SVM classifier.
        creating_dataset (bool, optional): Whether to create a dataset. Defaults to False.
        directory_name (str, optional): The name of the directory. Defaults to "".
    
    Returns:
        list: A list of soccer players.
    """

    global counter
    contours, hierarchy = cv2.findContours(campo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    soccer_players : list[Player] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 50:
            player_img = image[y : y + h, x : x + w]
            binary_player_img = campo[y : y + h, x : x + w]
            masked_image = cv2.bitwise_and(player_img, player_img, mask=binary_player_img)
            position = np.array([x + w/2,y + h/2])
            if creating_dataset :
                if w <= h/2:
                        cv2.imwrite(f"datasets/{directory_name}/masked_image{counter}.png", masked_image)
                        counter = counter + 1
                continue

            label = predict(svm_classifier,masked_image)
            player = Player(label,position)
            soccer_players.append(player)
            if label == [0]:
                cv2.rectangle(image, (x, y), (x + w, y + h),  (0, 255, 0), 2)
            elif label == [1] :
                cv2.rectangle(image, (x, y), (x + w, y + h),  (0, 0, 255), 2)
            elif label == [2]:
                cv2.rectangle(image, (x, y), (x + w, y + h),  (255, 0, 0), 2)
            elif label == [3]:
                cv2.rectangle(image, (x, y), (x + w, y + h),  (255, 255, 255), 2)
            elif label == [4]:
                cv2.rectangle(image, (x, y), (x + w, y + h),  (255, 255, 0), 2)
            elif label == [5]:
                cv2.rectangle(image, (x, y), (x + w, y + h),  (255, 0, 0), 2)

    return soccer_players



def deforma_immagine(img, punti,soccer_players):

    """
    Deforms the image based on the given points and updates the position of the soccer players.

    Args:
        img (numpy.ndarray): The input image.
        punti (list): The list of points.
        soccer_players (list): The list of soccer players.
    
    Returns:
        numpy.ndarray: The deformed image.
    """

    altezza, larghezza = img.shape[:2]
    punti_iniziali = np.float32(punti)
    punti_finali = np.float32([[0, 0], [larghezza, 0], [0, altezza], [larghezza, altezza]])
    transform_matrix = cv2.getPerspectiveTransform(punti_iniziali, punti_finali)
    img_deformata = cv2.warpPerspective(img, transform_matrix, (larghezza, altezza))

    for player in soccer_players:
        x_input, y_input = player.image_position
        point_input = np.array([x_input, y_input, 1])
        point_output_homogeneous = np.dot(transform_matrix, point_input)
        point_output = point_output_homogeneous[:2] / point_output_homogeneous[2]
        x_output, y_output = point_output
        player.transformed_position = np.array([x_output, y_output])

    return img_deformata

