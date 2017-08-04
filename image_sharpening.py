from PIL import ImageFilter
from PIL import Image


def image_sharpening(image):
    ''' image is taken as a parameter to input all images to code'''

    #Read image
    im = Image.open(image)   #'100037.jpg' '100018.jpg'
    #Display image
    #Applying a filter to the image
    im_sharp = im.filter(ImageFilter.SHARPEN)
    #Saving the filtered image to a new file
    im_sharp.save('tmp_image_sharpened.jpg', 'JPEG')

    #Splitting the image into its respective bands, i.e. Red, Green,
    #and Blue for RGB
    r,g,b = im_sharp.split()

    #Viewing EXIF data embedded in image
    exif_data = im._getexif()
    exif_data
    newimage = im
    return newimage
