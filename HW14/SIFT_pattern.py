import cv2

src = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
src_small = cv2.imread("lena_small.png", cv2.IMREAD_GRAYSCALE)

sift_object = cv2.SIFT.create(1000)
points, descriptors = sift_object.detectAndCompute(src, None)
points_small, descriptors_small = sift_object.detectAndCompute(src_small, None)

matcher = cv2.DescriptorMatcher.create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
matches = list(matcher.match(descriptors, descriptors_small))
matches.sort(key=lambda x: x.distance)
size = len(matches)
matches = (matches[:int(size*0.2)])

imMatches = cv2.drawMatches(src, points, src_small, points_small, matches, None)
cv2.imwrite("matches.jpg", imMatches)