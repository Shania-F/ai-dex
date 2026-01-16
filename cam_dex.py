import cv2
import tensorflow as tf
model = tf.keras.models.load_model('PokedeepVGG.hdf5')

from PIL import Image, ImageOps
import numpy as np
from keras.preprocessing import image

def predict_pokemon(p_name):
    img_width, img_height = 224, 224
    img = image.load_img(p_name, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    print(img.shape)
    print("break")
    img = np.expand_dims(img, axis = 0)
    print(img.shape)
    prediction = model.predict(img)
    pok= ['Blaziken','Charizard','Eevee','Empoleon','Haunter','Golbat', 'Jigglypuff','Machamp','Meowth', 'Onix','Pidgeot','Pikachu','Scyther','Snorlax','Squirtle','Venusaur']
    return pok[np.argmax(prediction)]

cap=cv2.VideoCapture(0)
scan=0

while(True):
    ret, scan=cap.read()
    cv2.imshow('Scanning...',scan)
    key=cv2.waitKey(1)
    if(key==27):
        break        
cv2.imwrite('scan.jpg',scan)

cv2.imshow("Scan",scan)
p_name='scan.jpg'
prediction = predict_pokemon(p_name)
print('This is your Pokemon: ', prediction)

cv2.destroyAllWindows()
cap.release()
