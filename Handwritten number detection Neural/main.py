
# coding: utf-8

# In[2]:


#import required modules
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from PIL import Image
import numpy as np
import os


# In[4]:


seed = 7
np.random.seed(seed)


# In[5]:


# load data to train
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[6]:


# Reshaping the batch of data  (batch, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')


# In[7]:


# this function takes image label, image directory, features data, labels data as input and then checks whether it is png file or not
def load_images_to_data(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    for file in list_of_files:
        image_file_name = os.path.join(image_directory, file)
        if ".png" in image_file_name:
            img = Image.open(image_file_name).convert("L")
            img = np.resize(img, (28,28,1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,28,28,1)
            features_data = np.append(features_data, im2arr, axis=0)
            label_data = np.append(label_data, [image_label], axis=0)
    return features_data, label_data


# In[8]:


# Now let’s give our own images directories to load them to existing dataset
X_train, y_train = load_images_to_data('1', 'data/mnist_data/train/1', X_train, y_train)
X_test, y_test = load_images_to_data('1', 'data/mnist_data/validation/1', X_test, y_test)


# In[9]:


# normalizing data
X_train/=255
X_test/=255


# In[10]:


# categorizing labels
number_of_classes = 10
y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)


# In[11]:


# creating test model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))


# In[12]:


# Compile test model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# In[13]:


# Fit the model to evaluate
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=7, batch_size=200)


# In[14]:


# Save the model
# model.save('models/mnistCNN.h5')


# In[15]:


# Evaluation of the datasets 
metrics = model.evaluate(X_test, y_test, verbose=0)
print("Metrics(Test loss & Test Accuracy): ")
print(metrics)


# In[16]:


img = Image.open('data/mnist_data/validation/1/1_2.png').convert("L")
img = np.resize(img, (28,28,1))
im2arr = np.array(img)
im2arr = im2arr.reshape(1,28,28,1)


# In[17]:


#print the predicted number
y_pred = model.predict_classes(im2arr)
print(y_pred)

