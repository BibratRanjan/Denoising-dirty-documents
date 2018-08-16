from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
import numpy as np
import cv2
import glob
import os

train_images =  sorted(glob.glob(os.path.join('denoising_docs', 'train', '*.png')))
gt_images =  sorted(glob.glob(os.path.join('denoising_docs', 'train_cleaned', '*.png')))

#Can include the early clipping here but that is second priority.

assert len(train_images)==len(gt_images)

x_train = []
for train_image in train_images:
    img = cv2.imread(train_image)
    img = cv2.resize(img, (240, 320))
    x_train.append(img)

x_train = np.array(x_train)

x_gt = []
for gt_image in gt_images:
    img = cv2.imread(gt_image)
    img = cv2.resize(img, (240, 320))
    x_gt.append(img)

x_gt = np.asarray(x_gt)

x_train = x_train.astype('float32') / 255.
x_gt = x_gt.astype('float32') / 255.

cv2.imshow('input', x_gt[1])
cv2.waitKey(0)
cv2.imshow('input', x_train[1])
cv2.waitKey(0)

input_img = Input(shape=(320, 240, 3))  # adapt this if using `channels_first` image data format
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

"""at this point the representation is (4, 4, 8) i.e. 128-dimensional"""

#x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
early_stopper = EarlyStopping(patience=5)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = autoencoder.fit(x_train, x_gt,
                epochs=150,
                batch_size=128,
                shuffle=True,
#                    validation_data=(x_test_noisy, x_test),
                callbacks=[early_stopper])

autoencoder.save('autoencoder1.h5')


import matplotlib.pyplot as plt
acc = history.history['acc']
#val_acc = history.history['val_acc']
loss = history.history['loss']
#val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
#plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
