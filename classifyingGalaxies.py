from PIL import Image
img= Image.open("100018.jpg")
img.save('test_picture.jpg')
im = Image.open("100037.jpg")
im.save("alteredimg.png")
image= "100037.jpg"
def fraction_black_pixels(image):
   image_dimensions_in_pixels=image.size[0]*image.size[1]
   colors=image.convert('RGB').getcolors(image_dimensions_in_pixels) #converts image to greyscale then gets colors
   n_pixels_closeto_black=0
   for pixel_count,color in colors:
       if color/255>0.90:
           n_pixels_closeto_black+=pixel_count
   return n_pixels_closeto_black/image_dimensions_in_pixels

assets = image.assets(image)
for a in assets:
   f=fraction_black_pixels(a.get_asset_image().image)
   print("Blackness in image:",f)
   display(a.get_asset_image().image)