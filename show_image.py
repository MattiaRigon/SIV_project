import cv2
import numpy as np

def show_image(output_img):
	cv2.imshow("Output Image", output_img.astype(np.uint8))
	while True:
		key = cv2.waitKey(1) & 0xFF

		# Break the loop if the 'q' key is pressed or the window is closed
		if key == ord('q') or cv2.getWindowProperty("Output Image", cv2.WND_PROP_VISIBLE) < 1:
			break