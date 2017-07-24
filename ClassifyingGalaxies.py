from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
import numpy as np
import cv2

def hasSatelliteTrail(image):
    '''checks if image has satellite trail going through, no galaxy
    '''
    img = Image.open(image)
    f = np.asarray(img).astype(float)
    img = Image.fromarray(f.astype(np.uint8))
    for i in range(3):
        img = img.filter(ImageFilter.SHARPEN)

    img.save('sharpenedimg.png')
    img = cv2.imread('sharpenedimg.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    minLineLength =1000
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    counter = 0
    if lines != None:
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
            counter += 1
        return image, str(counter), "does not pass"
    else:
        return image, "pass"
    cv2.imwrite('houghlinesreturnimage.jpg',img)
    print(lines[0])


result = hasSatelliteTrail('example_lightstreak.jpg')
print(result)
#blurred_f = ndimage.gaussian_filter(f, 3)
#
#filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
#
#alpha = 30
#sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
#
#plt.figure(figsize=(12, 4))
#
#plt.subplot(131)
#plt.imshow(f)
#plt.axis('off')
#plt.subplot(132)
#plt.imshow(blurred_f)
#plt.axis('off')
#plt.subplot(133)
#plt.imshow(sharpened)
#plt.axis('off')
#
#plt.tight_layout()
#plt.savefig('galaxytest.png')#