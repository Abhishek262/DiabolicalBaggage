
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.preprocessing.image import load_img

from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

from keras.models import Model
from keras.optimizers import Adam

base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation


model=Model(inputs=base_model.input,outputs=preds)


for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True


model.load_weights('my_model.h5')

for element in ['nogun.N0000410.jpg','nogun.N0000411.jpg','nogun.N0000412.jpg','nogun.N0000008.jpg','nogun.N0000009.jpg','nogun.N0000010.jpg','nogun.N0000011.jpg','nogun.N0000012.jpg','nogun.N0000013.jpg'] : 
    
    image_pred = load_img(element, target_size=(224, 224))
    image_pred = img_to_array(image_pred)
    image_pred = image_pred.reshape((1, image_pred.shape[0], image_pred.shape[1], image_pred.shape[2]))
    print(element)
    
    x = model.predict(image_pred)
    print(x)
    print("")




