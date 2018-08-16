import numpy as np
from keras.models import Model
from keras.datasets import mnist
import cv2
from keras.models import load_model
from sklearn.metrics import label_ranking_average_precision_score
import time
import glob
import os
import matplotlib.pyplot as plt

test_images =  sorted(glob.glob(os.path.join('denoising_docs', 'test', '*.png')))
temp = sorted(os.listdir(os.path.join('denoising_docs', 'test')))
image_nums = []
for image_num in temp:
    image_nums.append(image_num.split('.')[0])

x_test = []
for test_image in test_images:
    img = cv2.imread(test_image)
    img = cv2.resize(img, (240, 320))
    x_test.append(img)

x_test = np.array(x_test)

x_test = x_test.astype('float32') / 255.

print('Loading model :')
t0 = time.time()
# Load previously trained autoencoder
autoencoder = load_model('autoencoder1.h5')
t1 = time.time()
print('Model loaded in: ', t1-t0)

def plot_denoised_images():
    denoised_images = autoencoder.predict(x_test)
    
    assert len(image_nums)==len(denoised_images)
    for index in range(len(image_nums)):
        name = os.path.join('denoised_test_imgs', image_nums[index] + '.png')
        plt.imsave(name, denoised_images[index])
    
    test_img = x_test[4]
    resized_test_img = cv2.resize(test_img, (240, 320))
    cv2.imshow('input', resized_test_img)
    cv2.waitKey(0)
    
    output = denoised_images[4]
    resized_output = cv2.resize(output, (240, 320))
    cv2.imshow('output', resized_output)
    cv2.waitKey(0)
    
plot_denoised_images()