#!python

import csv
import matplotlib.image as mpimg
import numpy as np
import os

DATA_DIR='./data2'

def readDrivingData(path):
    csv_lines=[]
    with open(os.path.join(DATA_DIR, "driving_log.csv")) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            csv_lines.append(line)

    #for l in csv_lines:
    #    print(l)

    img_shape = None
    images = []
    steering_angles = []
    for line in csv_lines:
#        center_img_path, left_img_path, right_img_path = line[0:3]
        for idx in range(3):
            orig_img_path = line[idx]
            local_img_path = os.path.join(DATA_DIR, "IMG", os.path.basename(orig_img_path))
            image = mpimg.imread(local_img_path)
            if img_shape is None:
                img_shape = image.shape
            images.append(image)
            angle = float(line[3])
            if idx == 1:
                # This an image from the left camera - increase the steering angle
                angle += 0.2
            elif idx == 2:
                # This an image from the right camera - decrease the steering angle
                angle -= 0.2
            steering_angles.append(angle)
            #print("path:", local_img_path, "  shape:", image.shape, "  steering angle:", angle)

    return np.array(images), np.array(steering_angles), img_shape

def augmentDrivingData(X_train, y_train):
    X_aug = []
    y_aug = []

    for X, y in zip(X_train, y_train):
        X_aug.append(np.fliplr(X))
        y_aug.append(-y)

    return np.array(X_aug), np.array(y_aug)



X_drive, y_drive, img_shape = readDrivingData(DATA_DIR)
print("X_drive.shape:", X_drive.shape, "  y_drive.shape:", y_drive.shape)

X_aug, y_aug = augmentDrivingData(X_drive, y_drive)
print("X_aug.shape:", X_aug.shape, "  y_aug.shape:", y_aug.shape)

#from PIL import Image
#img = Image.fromarray(X_drive[0], 'RGB')
#img.show()
#img2 = Image.fromarray(X_aug[0], 'RGB')
#img2.show()


#X_train = np.concatenate([X_drive, X_aug])
#y_train = np.concatenate([y_drive, y_aug])
X_train = np.concatenate([X_aug])
y_train = np.concatenate([y_aug])
assert(len(X_train) == len(y_train))



from keras.models import Sequential
from keras.layers import Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=img_shape))
model.add(Cropping2D(cropping=((70, 15), (0, 0))))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())#pool_size=(2, 2)))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())#pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(84, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))#, activation='softmax'))


model.compile(optimizer="adam", loss="mse")
hist_obj = model.fit(X_train, y_train, validation_split=0.2, nb_epoch=8, verbose=1, shuffle="True")
model.save("model.h5")

print(hist_obj.history.keys())

import os

if 'DISPLAY' in os.environ:
    import matplotlib.pyplot as plt

    plt.plot(hist_obj.history['loss'])
    plt.plot(hist_obj.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

