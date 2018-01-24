# 202598
# 197012
from PIL import Image
import face_recognition
import os


list = os.listdir("./")
for i in range(0, len(list)):
    imgName = os.path.basename(list[i])
    if (os.path.splitext(imgName)[1] != ".jpg"): continue
    print(imgName)

    image = face_recognition.load_image_file(imgName)

    face_locations = face_recognition.face_locations(image)

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        #print('top:{} right:{} bottom:{} left:{}'.format(top, right, bottom, left))
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        width = right - left
        height = bottom - top
        if (width > height):
            right -= (width - height)
        elif (height > width):
            bottom -= (height - width) 
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image = pil_image.resize((64, 64), Image.ANTIALIAS)
        pil_image.save('../uncompletion_image/%s'%imgName)