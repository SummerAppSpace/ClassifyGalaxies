from PIL import ImageFilter
from PIL import Image, ImageChops

#Read image
im = Image.open('100042.jpg')   #'100037.jpg' '100018.jpg'
#Display image
#Applying a filter to the image
im_sharp = im.filter( ImageFilter.SHARPEN )
#Saving the filtered image to a new file
im_sharp.save( 'image_sharpened3.jpg', 'JPEG' )

#Splitting the image into its respective bands, i.e. Red, Green,
#and Blue for RGB
r,g,b = im_sharp.split()

#Viewing EXIF data embedded in image
exif_data = im._getexif()
exif_data
im.save('fixedimg3.png')
residual = ImageChops.subtract(im,im)