import cv2
import numpy as np

def linear_operate(path, a, b):
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)			# read images in the gray mode
	img_changed = (img * a + b).astype(np.uint8)			# do the linear operation
	imgs = np.hstack([img,img_changed])				# pack the original and changed images to display side by side
	cv2.imshow("linear operation: f_out = a * f_in + b", imgs)	# display
	cv2.waitKey(0)							# wait to close the window
	cv2.imwrite("HW1.1_result.png", imgs)				# save

if __name__ == "__main__":
	linear_operate(path="./pic_gray.png", a=1.2, b=-20)
