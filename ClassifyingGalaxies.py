#q1
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import csv
from sklearn.cluster import KMeans
import glob
from sklearn import preprocessing
import zipfile
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

def getRGBPixels(image):
    '''
    This is going to get the RGB value of every pixel in the images.
    '''
    image = Image.open(image)
    rgbimage = image.convert('RGB') #converts image to rgb colors
    colors = rgbimage.getdata()
    image.close()
    rgbimage.close()
    return colors

def getAllImagesRGB(images):
    rgblist = []
    for i in images:
        rgblist.append(getRGBPixels(i))
    return np.asarray(rgblist)


def training(xtrain, xtest, nneighbors):
    #code for the training solutions columns we want for this specific q
    columns = defaultdict(list) # each value in each column is appended to a list
    
    with open('training_solutions_rev1.csv') as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value
                columns[k].append(float(v)) # append the value into the appropriate list
                                 # based on column name k
    newlist = []
    biglist = []
    finallist = []
    yopt = [columns['Class1.1'], columns['Class1.2'], columns['Class1.3']]
    columnlist = ['Class1.1', 'Class1.2', 'Class1.3']
    for i in range(len(columns['Class1.1'])):
        for entry in yopt:
            newlist.append(entry[i])
        biglist.append(newlist)  
        newlist = []
    testcount = 0
    for entry in biglist:
        biggestval = (max(entry[:]))
        testcount += 1
        counter = 0
        for i in range(3):
            if entry[i] == biggestval and counter == 0:
                entry[i] = 1
                counter += 1
                finallist.append(columnlist[i])
            else:
                entry[i] = 0
                
    X = xtrain[:25]
    y = finallist[:25]

    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X, y) 
    print(clf.predict(rsubblist[0][0]))
    print(clf.predict_proba(rsubblist[0][0]))
    
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


    #h = .02  # step size in the mesh
    #
    ## Create color maps
    #cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    #cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    #
    #for weights in ['uniform', 'distance']:
    #    # we create an instance of Neighbours Classifier and fit the data.
    #    clf = neighbors.KNeighborsClassifier(n_neighbors=nneighbors, weights=weights)
    #    clf.fit(X, y)
    #
    #    # Plot the decision boundary. For that, we will assign a color to each
    #    # point in the mesh [x_min, x_max]x[y_min, y_max].
    #    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                         np.arange(y_min, y_max, h))
    #    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #
    #    # Put the result into a color plot
    #    Z = Z.reshape(xx.shape)
    #    plt.figure()
    #    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    #
    #    # Plot also the training points
    #    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    #    plt.xlim(xx.min(), xx.max())
    #    plt.ylim(yy.min(), yy.max())
    #    plt.title("3-Class classification (k = %i, weights = '%s')"
    #          % (n_neighbors, weights))
#
    #    plt.show()


    return('hi')

if __name__ == "__main__":
    imagelist = []
    zip_ref = zipfile.ZipFile('images_training_rev1.zip', 'r')
    zip_ref.extractall("Downloads")
    zip_ref.close()
    counter = 0
    for file in glob.glob("Downloads\images_training_rev1\*.jpg"):
        im = getRGBPixels(file)
        imagelist.append(im)
        counter += 1
        if counter >= 25: 
            break
    imagelist = np.asarray(imagelist)
    totalrlist = []
    rsubblist = []
    rlist = []
    glist = []
    blist = []
    rval = 0
    bval = 0
    for i in range(len(imagelist)):
        rval = 0
        bval = 0
        for j in range(len(im)):
            #rlist.append(imagelist[i][j][0])
            #glist.append(imagelist[i][j][1])
            #blist.append(imagelist[i][j][2])
            rval+=imagelist[i][j][0]
            bval+=imagelist[i][j][2]
        avgr = rval/(len(im))
        avgb = bval/(len(im))
        blist.append(avgb)
        rlist.append(avgr)
        rsubblist.append([avgr-avgb])
    
    #phrase your cv is phased the way they want
    
    imagelist2 = []
    zip_ref2 = zipfile.ZipFile('images_test_rev1.zip', 'r')
    zip_ref2.extractall("Downloads")
    zip_ref2.close()
    counter2 = 0
    for file2 in glob.glob("Downloads\images_test_rev1\*.jpg"):
        im2 = getRGBPixels(file2)
        imagelist2.append(im2)
        counter2 += 1
        if counter2 >= 25: 
            break
    imagelist2 = np.asarray(imagelist2) 
    rlist2 = []
    glist2 = []
    blist2 = []
    rsubblist2=[]
    for i in range(len(imagelist2)):
        rval2 = 0
        bval2 = 0
        for j in range(len(imagelist2)):
            #rlist2.append(imagelist2[i][j][0])
            #glist2.append(imagelist2[i][j][1])
            #blist2.append(imagelist2[i][j][2])
            rval2+=imagelist2[i][j][0]
            bval2+=imagelist2[i][j][2]
        avgr2 = rval2/(len(im))
        avgb2 = bval2/(len(im))
        blist2.append(avgb2)
        rlist2.append(avgr2)
        rsubblist2.append(avgr2-avgb2)
    
    res = training(rsubblist, rsubblist2, 3)
    print(res)










#from scipy import ndimage
#import matplotlib.pyplot as plt
#from PIL import Image
#from PIL import ImageFilter
#import numpy as np
#import cv2
#import math
#
#def sharpenImage(image):
#    img = Image.open(image)
#    for i in range(3):
#        img = img.filter(ImageFilter.SHARPEN)
#    return img
#
#
#def hasSatelliteTrail(image):
#    '''checks if image has satellite trail going through, no galaxy
#    '''
#    img = sharpenImage(image)
#    img.save('sharpenedimg.png')
#    img = cv2.imread('sharpenedimg.png')
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    edges = cv2.Canny(gray,50,150,apertureSize = 3)
#
#    minLineLength =1000
#    maxLineGap = 10
#    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#    counter = 0
#    if lines != None:
#        for x1,y1,x2,y2 in lines[0]:
#            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#            counter += 1
#        return image, str(counter), "does not pass"
#    else:
#        return image, "pass"
#    cv2.imwrite('houghlinesreturnimage.jpg',img)
#    print(lines[0])
#
#def fraction_color_pixels(rgbtuple, image):
#    # Greyscale ranges from 0 (black) to 255 (white).
#    image = Image.open(image)
#    image_dimensions_in_pixels=image.size[0]*image.size[1]
#    colors=image.convert('RGB').getcolors(image_dimensions_in_pixels) #converts image to greyscale then gets colors
#    n_pixels_closeto_value=0
#    for pixel_count,color in colors:
#        print(color)
#        #r = g = b = False
#        #colorlist = False
#        if (color[0] >= 215 and color[0] <= 239) and (color[1] >= 136 and color[1] <= 208) and (color[2] >= 40 and color[2] <= 169): # this is checking if it's close to the value
#            n_pixels_closeto_value+=1
#    image.save("pleasework.jpg")
#    print(image_dimensions_in_pixels)
#
#
#    return n_pixels_closeto_value/image_dimensions_in_pixels
#
#
#img2 = cv2.imread('119146.jpg',0)
#edges = cv2.Canny(img2,60,100)
#img3 = Image.fromarray(edges)
#img3 = img3.resize((500, 500))
#img3.save("doesthiswork.png")
#
#
#img2 = Image.open('119146.jpg')
#blue=[136, 141, 169]
#orange=[224, 163, 101]
#test = fraction_color_pixels(blue, '119146.jpg')
#test = fraction_color_pixels(orange, '100018.jpg')
#
#print(str(test) + str("test"))
#
#
#plt.subplot(121)
#plt.imshow(img2,cmap = 'gray')
#plt.title('Original Image')
#plt.xticks([])
#plt.yticks([])
#plt.subplot(122)
#plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
#plt.savefig('cannytest.png')
#
#result = hasSatelliteTrail('example_lightstreak.jpg')
#print(result)
#
##R: 136 G: 141 B: 169 blue (136, 141, 169)
##R: 224 G: 163 B: 101 orange (224, 163, 101)
#
##pass a series of tests for spiral - for not black, has canny curve edges + is certain part blue and not this orange
##elliptical-for the not black parts, is certain part orange and not more than this blue
##flowchart
