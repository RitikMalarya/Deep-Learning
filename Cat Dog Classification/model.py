#!/usr/bin/env python
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt


# In[1]:


import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


print(tf.__version__)


# ### Data Preprocessing

# In[3]:


#Processing the training set 
train_datagen = ImageDataGenerator(rescale= 1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')


# In[4]:


# processing the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                           target_size=(64,64),
                                           batch_size=32,
                                           class_mode='binary')


# ### Fitting CNN Model

# In[5]:


cnn = keras.models.Sequential([
    keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]),
    keras.layers.MaxPooling2D(pool_size=2,strides=2),
    keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2,strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128,activation='relu'),
    keras.layers.Dense(units=1,activation='sigmoid')
])


cnn.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')


cnn.fit(x=training_set,validation_data=test_set,epochs=10)


# ### Making Prediction on single image

# In[12]:


import numpy as np
from PIL import Image
from keras.preprocessing import image
test_img = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img,axis=0)

# Prediction
result = cnn.predict(test_img)
if result[0][0]==0:
    prediction='Cat'
else:
    prediction='Dog'
    
print(prediction)





# In[ ]:




