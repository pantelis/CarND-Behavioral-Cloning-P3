import csv
import cv2
import os
import numpy as np
import time
from keras.layers import Cropping2D
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.normalization import BatchNormalization

# driving the car forward in track 1
root_data_dir = '../datasets/udacity-car-sim-data/'
drive_dirs = ['final-forward-track1/', 'recovery-track1/', 'recovery-track1-additional']

# number of cameras to use (max 3)
# camera indeces (0,1,2) = (center, left, right)
num_cameras = 1

images = []
augmented_images = []
measurements = []
augmented_measurements = []
for directory in drive_dirs:
    print(directory)
    with open(os.path.join(root_data_dir, directory, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # use all available cameras
            for camera in range(num_cameras):
                # the following line uses separator for simulator data collected in windows
                # filename = line[0].split('\\')[-1]
                # the following line uses separator for simulator data collected in OSX
                filename = line[camera].split('/')[-1]
                # image is in BGR color space (default of cv.imread)
                image_bgr = cv2.imread(os.path.join(root_data_dir, directory, 'IMG/', filename))
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                images.append(image_rgb)
                measurements.append(float(line[3]))

# augmentation
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement)
    augmented_measurements.append(measurement*(-1.0))

print('Number of original images:', len(images))
print('Number of augmented images:', len(augmented_images))


# for img in images:
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)

def nvidia_model():

    model = Sequential()

    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(BatchNormalization(epsilon=0.001, axis=3, input_shape=(90, 320, 3)))

    model.add(Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    return model

model = nvidia_model()

model.compile(loss='mse', optimizer='adam')

# Training Pipeline
beginTime = time.time()

model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, epochs=5)
model.save('model.h5')

endTime = time.time()
print('Training time: {:5.2f}s'.format(endTime - beginTime))

import gc; gc.collect()