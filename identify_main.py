import cv2
from numpy import array
def picture_outlining(image):
    """
    Gives rough outline of an object in the picture in white and black pixels. Helps identify the shape of the galaxy, especially with identifying if they're spiral or eliptical
    """
    img = cv2.imread('100018.jpg') #make sure to change the name of the picture since it currently can only work with one picture
    img2 = array( 200  * (img[:,:,2] > img[:,:, 1]), dtype='uint8')
    edges = cv2.Canny(img2, 60, 50) #doesnt really change picture when numbers are changed
    newfile = 'edges' + str(image) + '.png'
    cv2.imwrite(newfile, edges)

    _, contours, _= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      # contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.imread(image, cv2.IMREAD_COLOR) #cv2.CV_LOAD_IMAGE_COLOR original #change to same name as the image above but with .jpg
    gray1 = cv2.Canny(img, 40, 50) #changing first number up reduces the amount of lines, changing second number down increases the amount of lines
    cv2.imwrite('gray1.png', gray1)
