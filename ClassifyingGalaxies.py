#9:43 25 images
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
    '''
    This gets the pixels of all the images into one array.
    '''
    rgblist = []
    for i in images:
        rgblist.append(getRGBPixels(i))
    return np.asarray(rgblist)


def traintest(xtrain, xtest, nneighbors, nresponses, filenames):
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
    #yopt = [columns['Class2.1'], columns['Class2.2']]
    #yopt = [columns['Class3.1'], columns['Class3.2']]
    #yopt = [columns['Class4.1'], columns['Class4.2']]
    #yopt = [columns['Class5.1'], columns['Class5.2'], columns['Class5.3'], columns['Class5.4'], ]
    #yopt = [columns['Class6.1'], columns['Class6.2']]
    #yopt = [columns['Class7.1'], columns['Class7.2'], columns['Class7.3']]
    #yopt = [columns['Class8.1'], columns['Class8.2'], columns['Class8.3'], columns['Class8.4'], columns['Class8.5'], columns['Class8.6'], columns['Class8.7']]
    #yopt = [columns['Class9.1'], columns['Class9.2'], columns['Class9.3']]
    #yopt = [columns['Class10.1'], columns['Class10.2'], columns['Class10.3']]
    #yopt = [columns['Class11.1'], columns['Class11.2'], columns['Class11.3'], columns['Class11.4'], columns['Class11.5'], columns['Class11.6']]



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
    clf.classes_
    probalist = []
    problist2 = []
    for i in range(len(xtest)):
        print(clf.predict(xtest[i]))
        probalist.append(clf.predict_proba(xtest[i])[0][0])
        problist2.append(clf.predict_proba(xtest[i])[0][1])

    
    with open('testml.csv', 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(xtest)):
            writer.writerow([filenames[i], str(probalist[i]), str(problist2[i])])
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


    #return('hi')

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
        if counter >= 6000: 
            break
    imagelist = np.asarray(imagelist)
    rsubblist = []
    avgrlist = []
    avgblist = []
    for i in range(len(imagelist)):
        rval = 0
        bval = 0
        for j in range(len(im)):
            test = imagelist[i][j][0]-imagelist[i][j][2]
            rval+=imagelist[i][j][0]
            bval+=imagelist[i][j][2]
        avgr = rval/(len(im))
        avgb = bval/(len(im))
        avgblist.append(avgb)
        avgrlist.append(avgr)
        rsubblist.append([avgr-avgb]) #takes the average red - blue of all pixels in training dataset
    
    filelist = []
    imagelist2 = []
    zip_ref2 = zipfile.ZipFile('images_test_rev1.zip', 'r')
    zip_ref2.extractall("Downloads")
    zip_ref2.close()
    counter2 = 0
    for file2 in glob.glob("Downloads\images_test_rev1\*.jpg"):
        im2 = getRGBPixels(file2)
        filelist.append(file2[27:33])
        imagelist2.append(im2)
        counter2 += 1
        if counter2 >= 6000: 
            break
    imagelist2 = np.asarray(imagelist2) 
    avgrlist2 = []
    avgblist2 = []
    rsubblist2=[]
    for i in range(len(imagelist2)):
        rval2 = 0
        bval2 = 0
        for j in range(len(im)):
            rval2+=imagelist2[i][j][0]
            bval2+=imagelist2[i][j][2]
        avgr2 = rval2/(len(im))
        avgb2 = bval2/(len(im))
        avgblist2.append(avgb2)
        avgrlist2.append(avgr2)
        rsubblist2.append([avgr2-avgb2]) #takes the average red - blue of all pixels in test dataset
    
    res = traintest(rsubblist, rsubblist2, 15, 3, filelist)
    print(res)