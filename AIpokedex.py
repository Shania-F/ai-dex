import tensorflow as tf
model = tf.keras.models.load_model('PokedeepVGG.hdf5')

from PIL import Image, ImageOps
import numpy as np
from keras.preprocessing import image
#from IPython.display import Image, display

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

p_name = input('Enter image name of pokemon:')
prediction = predict_pokemon(p_name)
print('This is your pokemon: ', prediction)
#display(Image(filename=p_name))
