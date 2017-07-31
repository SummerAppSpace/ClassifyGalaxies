import cv2
from numpy import array
img = cv2.imread('100018.jpg')
img2 = array( 200  * (img[:,:,2] > img[:,:, 1]), dtype='uint8')
edges = cv2.Canny(img2, 60, 50) #doesnt really change picture when numbers are changed
cv2.imwrite('edges1.png', edges)

_, contours, _= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

img = cv2.imread('100018.jpg', cv2.IMREAD_COLOR) #cv2.CV_LOAD_IMAGE_COLOR original
gray1 = cv2.Canny(img, 40, 50) #changing first number up reduces the amount of lines, changing second number down increases the amount of lines
cv2.imwrite('gray1.png', gray1)
