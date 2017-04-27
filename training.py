import csv
import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D

# driving the car forward in track 1
root_data_dir = '../datasets/'
drive_dirs = ['udacity-data-track1/']
#drive_dirs = ['reverse-track1/']#, 'forward-track2/', 'reverse-track2/']#, ]

images = []
augmented_images = []
measurements = []
augmented_measurements = []
for directory in drive_dirs:
    print(directory)
    with open(os.path.join(root_data_dir, directory, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # use the center image as a feature (index 0)
            # the following line uses separator for simulator data collected in windows
            filename = line[0].split('\\')[-1]
            #image = cv2.imread(os.path.join(root_data_dir, directory, 'IMG/', filename))
            image = cv2.imread(os.path.join(root_data_dir, directory, filename))
            #print(os.path.join(root_data_dir, directory, filename))
            images.append(image)
            measurements.append(float(line[3]))

# augmentation
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement)
    augmented_measurements.append(measurement*(-1.0))

print('Number of images:', len(images))

# for img in images:
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)

# Simple network
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')

