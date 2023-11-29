import cv2
import numpy as np

def add(path1, path2):
	img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
	img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)		# read the two images
	img_added = img1 + img2					# do the adding operation
	imgs = np.hstack([img1, img2, img_added])		# pack the images to display them side by side
	cv2.imshow("add_operation", imgs)			# display
	cv2.waitKey(0)
	cv2.imwrite("HW1.2_result.png", imgs)			# save

if __name__ == "__main__":
	add("./pic1.jpeg", "./pic2.jpeg")
