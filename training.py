import csv
import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.normalization import BatchNormalization

# driving the car forward in track 1
root_data_dir = '../datasets/udacity-car-sim-data/'
drive_dirs = ['final-forward-track1/']
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
            # use all available cameras
            for camera in range(3):
                # the following line uses separator for simulator data collected in windows
                # filename = line[0].split('\\')[-1]
                # the following line uses separator for simulator data collected in OSX
                filename = line[camera].split('/')[-1]
                image = cv2.imread(os.path.join(root_data_dir, directory, 'IMG/', filename))
                #image = cv2.imread(os.path.join(root_data_dir, directory, filename))
                #print(os.path.join(root_data_dir, directory, filename))
                images.append(image)
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


def simple_model():

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model


def nvidia_model():

    model = Sequential()

    model.add(BatchNormalization(epsilon=0.001, axis=3, input_shape=(160, 320, 3)))

    model.add(Conv2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    return model

#model = simple_model()
model = nvidia_model()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')