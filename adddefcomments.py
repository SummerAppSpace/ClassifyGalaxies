def getRGBPixels(image):
    '''
    This is going to get the RGB value of every pixel in the images.
    '''
    image = Image.open(image) #opens image
    rgbimage = image.convert('RGB') #converts image to rgb colors
    colors = rgbimage.getdata() #get rgb value of each pixel
    image.close() #closes image
    rgbimage.close() #closes coverted image
    return colors #returns list of rgv values of all pixels in 1 image
    
def getAllImagesRGB(images): 
    '''
    This gets the pixels of all the images into one array.
    '''
    rgblist = []
    for i in images: # runs all images through rgb values and gets color value
        rgblist.append(getRGBPixels(i))
    return np.asarray(rgblist) #returns as numpy array