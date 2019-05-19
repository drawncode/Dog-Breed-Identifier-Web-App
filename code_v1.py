
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import os
from glob import glob
import pickle
from tqdm import tqdm
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
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator



img_width, img_height = 224, 224
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)




model1 = InceptionV3(weights=None,include_top=False, input_shape=(224, 224, 3))


model1.load_weights('weights/model1.h5')

model2 = Sequential()
model2.add(GlobalAveragePooling2D(input_shape=(5,5,2048)))
model2.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
model2.add(Dropout(0.4))
model2.add(Dense(133, activation='softmax'))

# model2.summary()


model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])





model2.load_weights('weights/model2.hdf5')





path="output/"




images=glob(path+"*.jpg")
result_images=images
images=np.array(images)
image_tensors=paths_to_tensor(images).astype('float32')/255



bottleneck=[model1.predict(np.expand_dims(tensor, axis=0)) for tensor in image_tensors]




prediction = [np.argmax(model2.predict(feature)) for feature in bottleneck]


with open('dog_names', 'rb') as f:
    dog_names=pickle.load(f)



result_breeds=[dog_names[i] for i in prediction]
print(result_images)
print(result_breeds)



l=len(result_images)

fig=plt.figure(figsize=(7,7))
if l>3:
	columns = 3
else:
	columns=l
rows=math.ceil(l/columns)
# fig, ax = plt.subplots()
# [axi.set_axis_off() for axi in ax.ravel()]
for i in range(0, l):
	print(result_images[i])
	img = cv2.imread(result_images[i])
	# print(img.shape)
	fig.add_subplot(rows, columns, i+1)
	plt.title(result_breeds[i],fontdict = {'fontsize' : 10})
	plt.imshow(img)
	# plt.subplots_adjust(wspace=0.5)
	cur_axes = plt.gca()
	cur_axes.axes.get_xaxis().set_visible(False)
	cur_axes.axes.get_yaxis().set_visible(False)
plt.savefig("dogs_result.jpg")
# plt.savefig("temp.jpg")

img_1 = cv2.imread("dogs_result.jpg")
img_1=cv2.resize(img_1,(700,700))
img_2 = cv2.resize(cv2.imread("predictions.jpg"),(700,700))
f_img = np.zeros((700,1400,3))

f_img[:,:700,:]=img_2
f_img[:,700:1400,:]=img_1

cv2.imwrite('static/photo/dogs_result.jpg',f_img)

# with open('dog_names','wb') as f:
#     pickle.dump(dog_names,f)



