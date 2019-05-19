import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_files       
from keras.utils import np_utils
from glob import glob
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout,GlobalAveragePooling2D,AveragePooling2D
from keras.utils import layer_utils
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm




# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('train')
valid_files, valid_targets = load_dataset('valid')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("train/*/"))]
dog_breeds = len(dog_names)

img_width, img_height = 224, 224
def path_to_tensor(img_path):
	img = image.load_img(img_path, target_size=(img_width, img_height))
	x = image.img_to_array(img)
	return np.expand_dims(x, axis=0)
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


model1 = InceptionV3(weights=None,include_top=False, input_shape=(224, 224, 3))
# mode1.summary()
model1.load_weights('weights/model1.h5')

model2 = Sequential()
model2.add(GlobalAveragePooling2D(input_shape=(5,5,2048)))
model2.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
model2.add(Dropout(0.4))
model2.add(Dense(133, activation='softmax'))

# model2.summary()


model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('train',target_size=(224, 224),batch_size=20,class_mode='categorical')
valid_set = valid_datagen.flow_from_directory('valid',target_size=(224, 224),batch_size=22,class_mode='categorical')
# train_tensors=paths_to_tensor(train_files).astype('float32')/255
valid_tensors=paths_to_tensor(valid_files).astype('float32')/255
# train_bottleneck=[model1.predict(np.expand_dims(tensor, axis=0)) for tensor in train_tensors]
valid_bottleneck=[model1.predict(np.expand_dims(tensor, axis=0)) for tensor in valid_tensors]

# checkpointer = ModelCheckpoint(filepath='weights/model2.hdf5',verbose=1, save_best_only=True)

# model2.fit(train_bottleneck, train_targets, validation_data=(valid_bottleneck, valid_targets),epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
