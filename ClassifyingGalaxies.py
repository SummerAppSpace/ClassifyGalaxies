#9:43 25 images
#q1
import sys
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
def traintest(xtrain, xtest, nneighbors, filenames):
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
            if entry[i] == biggestval and counter == 0 and entry[i] > 0.20:
                entry[i] = 1
                counter += 1
                finallist.append(columnlist[i])
            else:
                entry[i] = 0
    X = xtrain
    y = finallist#[:25] #take out for christine 
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X, y) 
    clf.classes_
    problist11 = []
    problist12 = []
    problist13 = []
    for i in range(len(xtest)):
        problist11.append(clf.predict_proba(xtest[i])[0][0])
        problist12.append(clf.predict_proba(xtest[i])[0][1])
        if len(clf.predict_proba(xtest[i])[0])==3:
            problist13.append(clf.predict_proba(xtest[i])[0][2])
        else:
            problist13.append(0)
            
    problist11 = np.asarray(problist11)
    problist12 = np.asarray(problist12)
    problist13 = np.asarray(problist13)


    #q2
    yopt = [columns['Class2.1'], columns['Class2.2']]
    columnlist = ['Class2.1', 'Class2.2']
    newlist=[]
    biglist=[]
    finallist=[]
    for i in range(len(columns['Class2.1'])):
        for entry in yopt:
            newlist.append(entry[i])
        biglist.append(newlist)  
        newlist = []
    testcount = 0
    for entry in biglist:
        biggestval = (max(entry[:]))
        testcount += 1
        counter = 0
        for i in range(2):
            if entry[i] == biggestval and counter == 0 and entry[i] > 0.20:
                entry[i] = 1
                counter += 1
                finallist.append(columnlist[i])
            else:
                entry[i] = 0
                
    X = xtrain
    y = finallist#[:25] #take out for christine
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X, y) 
    clf.classes_
    problist21 = []
    problist22 = []
    for i in range(len(xtest)):
        problist21.append(clf.predict_proba(xtest[i])[0][0])
        problist22.append(clf.predict_proba(xtest[i])[0][1])
        
    problist21 = np.asarray(problist21)
    problist22 = np.asarray(problist22)
    
    problist21 = problist12*problist21
    problist22 = problist12*problist22
    
    
    #q7
    newlist = []
    biglist = []
    finallist = []
    yopt = [columns['Class7.1'], columns['Class7.2'], columns['Class7.3']]
    columnlist = ['Class7.1', 'Class7.2', 'Class7.3']
    for i in range(len(columns['Class7.1'])):
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
            if entry[i] == biggestval and counter == 0 and entry[i] > 0.20:
                entry[i] = 1
                counter += 1
                finallist.append(columnlist[i])
            else:
                entry[i] = 0
    X = xtrain
    y = finallist#[:25] #take out for christine 
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X, y) 
    clf.classes_
    problist71 = []
    problist72 = []
    problist73 = []
    for i in range(len(xtest)):
        problist71.append(clf.predict_proba(xtest[i])[0][0])
        problist72.append(clf.predict_proba(xtest[i])[0][1])
        if len(clf.predict_proba(xtest[i])[0]) == 3:
            problist73.append(clf.predict_proba(xtest[i])[0][2])
        else:
            problist73.append(0)
    
    problist71 = np.asarray(problist71) * problist11
    problist72 = np.asarray(problist72) * problist11
    problist73 = np.asarray(problist73) * problist11
    
    #q6
    yopt = [columns['Class6.1'], columns['Class6.2']]
    columnlist = ['Class6.1', 'Class6.2']
    newlist=[]
    biglist=[]
    finallist=[]
    for i in range(len(columns['Class6.1'])):
        for entry in yopt:
            newlist.append(entry[i])
        biglist.append(newlist)  
        newlist = []
    testcount = 0
    for entry in biglist:
        biggestval = (max(entry[:]))
        testcount += 1
        counter = 0
        for i in range(2):
            if entry[i] == biggestval and counter == 0 and entry[i] > 0.20:
                entry[i] = 1
                counter += 1
                finallist.append(columnlist[i])
            else:
                entry[i] = 0
                
    X = xtrain
    y = finallist#[:25] #take out for christine
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X, y) 
    clf.classes_
    problist61 = []
    problist62 = []
    for i in range(len(xtest)):
        problist61.append(clf.predict_proba(xtest[i])[0][0])
        problist62.append(clf.predict_proba(xtest[i])[0][1])
        
    problist61 = np.asarray(problist61)
    problist62 = np.asarray(problist62)
    
    #q8
    yopt = [columns['Class8.1'], columns['Class8.2'], columns['Class8.3'],
            columns['Class8.4'], columns['Class8.5'], columns['Class8.6'], columns['Class8.7']]
    columnlist = ['Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 
                 'Class8.6', 'Class8.7']
    newlist=[]
    biglist=[]
    finallist=[]
    for i in range(len(columns['Class8.1'])):
        for entry in yopt:
            newlist.append(entry[i])
        biglist.append(newlist)  
        newlist = []
    testcount = 0
    for entry in biglist:
        biggestval = (max(entry[:]))
        testcount += 1
        counter = 0
        for i in range(2):
            if entry[i] == biggestval and counter == 0 and entry[i] > 0.20:
                entry[i] = 1
                counter += 1
                finallist.append(columnlist[i])
            else:
                entry[i] = 0
                
    X = xtrain
    y = finallist#[:25] #take out for christine
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X, y) 
    clf.classes_
    problist81 = []
    problist82 = []
    problist83 = []
    problist84 = []
    problist85 = []
    problist86 = []
    problist87 = []
    
    for i in range(len(xtest)):
        if len(clf.predict_proba(xtest[i])[0])==7: #can take out for christines but need it to run on mine
            problist81.append(clf.predict_proba(xtest[i])[0][0])
            problist82.append(clf.predict_proba(xtest[i])[0][1])
            problist83.append(clf.predict_proba(xtest[i])[0][2])
            problist84.append(clf.predict_proba(xtest[i])[0][3])
            problist85.append(clf.predict_proba(xtest[i])[0][4])
            problist86.append(clf.predict_proba(xtest[i])[0][5])
            problist87.append(clf.predict_proba(xtest[i])[0][6])
        else:
            problist81.append(0)
            problist82.append(0)
            problist83.append(0)
            problist84.append(0)
            problist85.append(0)
            problist86.append(0)
            problist87.append(0)
    
    problist81 = np.asarray(problist81) * problist61
    problist82 = np.asarray(problist82) * problist61
    problist83 = np.asarray(problist83) * problist61
    problist84 = np.asarray(problist84) * problist61
    problist85 = np.asarray(problist85) * problist61
    problist86 = np.asarray(problist86) * problist61
    problist87 = np.asarray(problist87) * problist61
    
    
    #q3
    yopt = [columns['Class3.1'], columns['Class3.2']]
    columnlist = ['Class3.1', 'Class3.2']
    newlist=[]
    biglist=[]
    finallist=[]
    for i in range(len(columns['Class3.1'])):
        for entry in yopt:
            newlist.append(entry[i])
        biglist.append(newlist)  
        newlist = []
    testcount = 0
    for entry in biglist:
        biggestval = (max(entry[:]))
        testcount += 1
        counter = 0
        for i in range(2):
            if entry[i] == biggestval and counter == 0 and entry[i] > 0.20:
                entry[i] = 1
                counter += 1
                finallist.append(columnlist[i])
            else:
                entry[i] = 0
                
    X = xtrain
    y = finallist#[:25] #take out for christine
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X, y) 
    clf.classes_
    problist31 = []
    problist32 = []
    for i in range(len(xtest)):
        if len(clf.predict_proba(xtest[i])[0]) == 2:
            problist31.append(clf.predict_proba(xtest[i])[0][0])
            problist32.append(clf.predict_proba(xtest[i])[0][1])
        else:
            problist31.append(0)
            problist32.append(0)
        
    problist31 = np.asarray(problist31)
    problist32 = np.asarray(problist32)
    
    problist31 = problist22*problist31
    problist32 = problist22*problist32
    
    
    #q4
    yopt = [columns['Class4.1'], columns['Class4.2']]
    columnlist = ['Class4.1', 'Class4.2']
    newlist=[]
    biglist=[]
    finallist=[]
    for i in range(len(columns['Class4.1'])):
        for entry in yopt:
            newlist.append(entry[i])
        biglist.append(newlist)  
        newlist = []
    testcount = 0
    for entry in biglist:
        biggestval = (max(entry[:]))
        testcount += 1
        counter = 0
        for i in range(2):
            if entry[i] == biggestval and counter == 0 and entry[i] > 0.20:
                entry[i] = 1
                counter += 1
                finallist.append(columnlist[i])
            else:
                entry[i] = 0
                
    X = xtrain
    y = finallist#[:25] #take out for christine
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X, y) 
    clf.classes_
    problist41 = []
    problist42 = []
    for i in range(len(xtest)):
        if len(clf.predict_proba(xtest[i])[0]) == 2:
            problist41.append(clf.predict_proba(xtest[i])[0][0])
            problist42.append(clf.predict_proba(xtest[i])[0][1])
        else:
            problist41.append(0)
            problist42.append(0)
        
    problist41 = np.asarray(problist41) * problist22
    problist42 = np.asarray(problist42) * problist22
    
    
    #q10
    newlist = []
    biglist = []
    finallist = []
    yopt = [columns['Class10.1'], columns['Class10.2'], columns['Class10.3']]
    columnlist = ['Class10.1', 'Class10.2', 'Class10.3']
    for i in range(len(columns['Class10.1'])):
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
            if entry[i] == biggestval and counter == 0 and entry[i] > 0.20:
                entry[i] = 1
                counter += 1
                finallist.append(columnlist[i])
            else:
                entry[i] = 0
    X = xtrain
    y = finallist#[:25] #take out for christine 
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X, y) 
    clf.classes_
    problist101 = []
    problist102 = []
    problist103 = []
    for i in range(len(xtest)):
        if len(clf.predict_proba(xtest[i])[0])==3:
            problist101.append(clf.predict_proba(xtest[i])[0][0])
            problist102.append(clf.predict_proba(xtest[i])[0][1])
            problist103.append(clf.predict_proba(xtest[i])[0][2])
        else:
            problist101.append(0)
            problist102.append(0)
            problist103.append(0)
            
    problist101 = np.asarray(problist101) * problist41
    problist102 = np.asarray(problist102) * problist41
    problist103 = np.asarray(problist103) * problist41
    
    #q11
    newlist = []
    biglist = []
    finallist = []
    yopt = [columns['Class11.1'], columns['Class11.2'], columns['Class11.3'], 
            columns['Class11.4'], columns['Class11.5'], columns['Class11.6']]
    columnlist = ['Class11.1', 'Class11.2', 'Class11.3', 
                  'Class11.4', 'Class11.5', 'Class11.6']
    for i in range(len(columns['Class11.1'])):
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
            if entry[i] == biggestval and counter == 0 and entry[i] > 0.20:
                entry[i] = 1
                counter += 1
                finallist.append(columnlist[i])
            else:
                entry[i] = 0
    X = xtrain
    y = finallist#[:25] #take out for christine 
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X, y) 
    clf.classes_
    problist111 = []
    problist112 = []
    problist113 = []
    problist114 = []
    problist115 = []
    problist116 = []
    for i in range(len(xtest)):
        if len(clf.predict_proba(xtest[i])[0])==6:
            problist111.append(clf.predict_proba(xtest[i])[0][0])
            problist112.append(clf.predict_proba(xtest[i])[0][1])
            problist113.append(clf.predict_proba(xtest[i])[0][2])
            problist114.append(clf.predict_proba(xtest[i])[0][3])
            problist115.append(clf.predict_proba(xtest[i])[0][4])
            problist116.append(clf.predict_proba(xtest[i])[0][5])
        else:
            problist111.append(0)
            problist112.append(0)
            problist113.append(0)
            problist114.append(0)
            problist115.append(0)
            problist116.append(0)
            
    problist111 = np.asarray(problist111) * problist41
    problist112 = np.asarray(problist112) * problist41
    problist113 = np.asarray(problist113) * problist41
    problist114 = np.asarray(problist114) * problist41
    problist115 = np.asarray(problist115) * problist41
    problist116 = np.asarray(problist116) * problist41
    
    
    
    #q11
    newlist = []
    biglist = []
    finallist = []
    yopt = [columns['Class5.1'], columns['Class5.2'], columns['Class5.3'], 
            columns['Class5.4']]
    columnlist = ['Class5.1', 'Class5.2', 'Class5.3', 
                  'Class5.4']
    for i in range(len(columns['Class11.1'])):
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
            if entry[i] == biggestval and counter == 0 and entry[i] > 0.20:
                entry[i] = 1
                counter += 1
                finallist.append(columnlist[i])
            else:
                entry[i] = 0
    X = xtrain
    y = finallist#[:25] #take out for christine 
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X, y) 
    clf.classes_
    problist51 = []
    problist52 = []
    problist53 = []
    problist54 = []
    for i in range(len(xtest)):
        if len(clf.predict_proba(xtest[i])[0])==4:
            problist51.append(clf.predict_proba(xtest[i])[0][0])
            problist52.append(clf.predict_proba(xtest[i])[0][1])
            problist53.append(clf.predict_proba(xtest[i])[0][2])
            problist54.append(clf.predict_proba(xtest[i])[0][3])
        else:
            problist51.append(0)
            problist52.append(0)
            problist53.append(0)
            problist54.append(0)
            
    problist51 = np.asarray(problist51) * problist41
    problist52 = np.asarray(problist52) * problist41
    problist53 = np.asarray(problist53) * problist41
    problist54 = np.asarray(problist54) * problist41
    
    #q9
    newlist = []
    biglist = []
    finallist = []
    yopt = [columns['Class9.1'], columns['Class9.2'], columns['Class9.3']]
    columnlist = ['Class9.1', 'Class9.2', 'Class9.3']
    for i in range(len(columns['Class11.1'])):
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
            if entry[i] == biggestval and counter == 0 and entry[i] > 0.20:
                entry[i] = 1
                counter += 1
                finallist.append(columnlist[i])
            else:
                entry[i] = 0
    X = xtrain
    y = finallist#[:25] #take out for christine 
    clf = KNeighborsClassifier(n_neighbors=nneighbors)
    clf.fit(X, y) 
    clf.classes_
    problist91 = []
    problist92 = []
    problist93 = []
    for i in range(len(xtest)):
        if len(clf.predict_proba(xtest[i])[0])==3:
            problist91.append(clf.predict_proba(xtest[i])[0][0])
            problist92.append(clf.predict_proba(xtest[i])[0][1])
            problist93.append(clf.predict_proba(xtest[i])[0][2])
        else:
            problist91.append(0)
            problist92.append(0)
            problist93.append(0)
            
    problist91 = np.asarray(problist91) * problist21
    problist92 = np.asarray(problist92) * problist21
    problist93 = np.asarray(problist93) * problist21

    
    
    
    
    rowcount = 0
    with open('testml10001500final.csv', 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        with open('Downloads/all_zeros_benchmark.csv') as f:
            reader = csv.reader(f) # read rows into a dictionary format
            for row in reader: # read a row as {column1: value1, column2: value2,...}
                if rowcount == 0:
                    writer.writerow([','.join(row)])
                    rowcount+= 1
        for i in range(len(xtest)):
            writer.writerow([filenames[i], str(problist11[i]), str(problist12[i]), str(problist13[i]), 
                             str(problist21[i]), str(problist22[i]), 
                             str(problist31[i]), str(problist32[i]),
                             str(problist41[i]), str(problist42[i]),
                             str(problist51[i]), str(problist52[i]),
                             str(problist53[i]), str(problist54[i]),
                             str(problist61[i]), str(problist62[i]), 
                             str(problist71[i]), str(problist72[i]), str(problist73[i]), 
                             str(problist81[i]), str(problist82[i]), str(problist83[i]), 
                             str(problist84[i]), str(problist85[i]), str(problist86[i]),
                             str(problist81[i]), str(problist82[i]), str(problist83[i]), 
                             str(problist87[i]), str(problist91[i]), str(problist92[i]), str(problist93[i]), 
                             str(problist101[i]), str(problist102[i]), 
                             str(problist103[i]), str(problist111[i]), str(problist112[i]), str(problist113[i]), 
                             str(problist114[i]), str(problist115[i]), str(problist116[i])])

    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #return('hi')
    
    
    
if __name__ == "__main__":
    try:
        begin = int(sys.argv[1])
        end = int(sys.argv[2])
    except IndexError:
        begin = 0
        end = 25
    
    imagelist = []
    zip_ref = zipfile.ZipFile('images_training_rev1.zip', 'r')
    zip_ref.extractall("Downloads")
    zip_ref.close()
    counter = 0
    for file in glob.glob("Downloads/images_training_rev1/*.jpg"):
        #counter+=1
        im = getRGBPixels(file)
        imagelist.append(im)
        #if counter >=25: #take out for christine
        #    break
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
    for file2 in glob.glob("Downloads/images_test_rev1/*.jpg"):
        counter2 += 1
        if counter2 >= begin:
            im2 = getRGBPixels(file2)
            filelist.append(file2[27:33])
            imagelist2.append(im2)
            if counter2 > end:
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
    
    res = traintest(rsubblist, rsubblist2, 15, filelist)
    print(res)