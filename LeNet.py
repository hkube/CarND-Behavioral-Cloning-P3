#!python

import csv
import matplotlib.image as mpimg
import numpy as np
import os
import sklearn
from click.core import batch

OUTER_CAM_ANGLE_DIFF=0.5
USE_GENERATOR=True

def readDrivingDataInfo(path):
    csv_lines=[]
    with open(os.path.join(path, "driving_log.csv")) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            csv_lines.append(line)

    #for l in csv_lines:
    #    print(l)
    return csv_lines

if not USE_GENERATOR:
    def readDrivingData(path):
        csv_lines = readDrivingDataInfo(path)
        img_shape = None
        images = []
        steering_angles = []
        for line in csv_lines:
    #        center_img_path, left_img_path, right_img_path = line[0:3]
            for idx in range(3):
                orig_img_path = line[idx]
                local_img_path = os.path.join(path, "IMG", os.path.basename(orig_img_path))
                image = mpimg.imread(local_img_path)
                if img_shape is None:
                    img_shape = image.shape
                images.append(image)
                angle = float(line[3])
                if idx == 1:
                    # This an image from the left camera - increase the steering angle
                    angle += 0.4
                elif idx == 2:
                    # This an image from the right camera - decrease the steering angle
                    angle -= 0.4
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

    if False:
        X_drive2, y_drive2, img_shape = readDrivingData('./data2')
        print("X_drive.shape:", X_drive2.shape, "  y_drive.shape:", y_drive2.shape)

        X_drive3, y_drive3, _ = readDrivingData('./data3')
        print("X_drive.shape:", X_drive3.shape, "  y_drive.shape:", y_drive3.shape)

        X_drive = np.concatenate([X_drive2, X_drive3])
        y_drive = np.concatenate([y_drive2, y_drive3])

    else:
        X_drive, y_drive, img_shape = readDrivingData('./data1')

    print("X_drive.shape:", X_drive.shape, "  y_drive.shape:", y_drive.shape)

    X_aug, y_aug = augmentDrivingData(X_drive, y_drive)
    print("X_aug.shape:", X_aug.shape, "  y_aug.shape:", y_aug.shape)

    #X_train = np.concatenate([X_drive, X_aug])
    #y_train = np.concatenate([y_drive, y_aug])
    X_train = np.concatenate([X_aug])
    y_train = np.concatenate([y_aug])
    assert(len(X_train) == len(y_train))

else:
    import sklearn
    from sklearn.model_selection import train_test_split

    def prepareDrivingDataSamples(dirs):
        if type(dirs) is not list:
            print("Putting", dirs, "into a list")
            dirs = [dirs]
        samples = []
        for d in dirs:
            print("Reading driving data from ", d)
            for line in readDrivingDataInfo(d):
                samples.append((os.path.join(d, "IMG", os.path.basename(line[0])), 'none', float(line[3])))
                samples.append((os.path.join(d, "IMG", os.path.basename(line[1])), 'none', float(line[3]) + OUTER_CAM_ANGLE_DIFF))
                samples.append((os.path.join(d, "IMG", os.path.basename(line[2])), 'none', float(line[3]) - OUTER_CAM_ANGLE_DIFF))
                samples.append((os.path.join(d, "IMG", os.path.basename(line[0])), 'flip', float(line[3])))
                samples.append((os.path.join(d, "IMG", os.path.basename(line[1])), 'flip', float(line[3]) + OUTER_CAM_ANGLE_DIFF))
                samples.append((os.path.join(d, "IMG", os.path.basename(line[2])), 'flip', float(line[3]) - OUTER_CAM_ANGLE_DIFF))
        return samples

    def generator(samples, batch_size=32):
        num_samples = len(samples)
        while True:
            sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    #print("file:", batch_sample[0], "  angle:", batch_sample[2])
                    img = mpimg.imread(batch_sample[0])
                    angle = batch_sample[2]
                    if batch_sample[1] == 'flip':
                        img = np.fliplr(img)
                        angle = -angle
                    images.append(img)
                    angles.append(angle)

                yield sklearn.utils.shuffle(np.array(images), np.array(angles))

    train_samples, valid_samples = train_test_split(prepareDrivingDataSamples(['./data1']), test_size=0.2)

    train_generator = generator(train_samples, batch_size=32)
    valid_generator = generator(valid_samples, batch_size=32)

    img_shape = mpimg.imread(train_samples[0][0]).shape
    print("img_shape:", img_shape)
    print("train_samples:", len(train_samples), "   valid_samples:", len(valid_samples))
#from PIL import Image
#img = Image.fromarray(X_drive[0], 'RGB')
#img.show()
#img2 = Image.fromarray(X_aug[0], 'RGB')
#img2.show()



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
if not USE_GENERATOR:
    hist_obj = model.fit(X_train, y_train, validation_split=0.2, nb_epoch=8, verbose=1, shuffle="True")
else:
    hist_obj = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                                   validation_data=valid_generator, nb_val_samples=len(valid_samples), \
                                   nb_epoch=8, verbose=1)
model.save("model_LeNet.h5")

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

