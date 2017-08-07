#!python

import csv
import matplotlib.image as mpimg
import numpy as np
import os
import sklearn
from click.core import batch

SIDECAM_BIAS=0.0
SIDECAM_FACTOR=3
NB_EPOCHS=5
DATA_DIRS=['./data2']

def readDrivingDataInfo(path):
    csv_lines=[]
    with open(os.path.join(path, "driving_log.csv")) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            csv_lines.append(line)

    #for l in csv_lines:
    #    print(l)
    return csv_lines

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
            samples.append((os.path.join(d, "IMG", os.path.basename(line[1])), 'none', float(line[3]) * SIDECAM_FACTOR + SIDECAM_BIAS))
            samples.append((os.path.join(d, "IMG", os.path.basename(line[2])), 'none', float(line[3]) * SIDECAM_FACTOR - SIDECAM_BIAS))
            samples.append((os.path.join(d, "IMG", os.path.basename(line[0])), 'flip', float(line[3])))
            samples.append((os.path.join(d, "IMG", os.path.basename(line[1])), 'flip', float(line[3]) * SIDECAM_FACTOR + SIDECAM_BIAS))
            samples.append((os.path.join(d, "IMG", os.path.basename(line[2])), 'flip', float(line[3]) * SIDECAM_FACTOR - SIDECAM_BIAS))
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

train_samples, valid_samples = train_test_split(prepareDrivingDataSamples(DATA_DIRS), test_size=0.2)

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
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(optimizer="adam", loss="mse")
hist_obj = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                               validation_data=valid_generator, nb_val_samples=len(valid_samples), \
                               nb_epoch=NB_EPOCHS, verbose=1)
model.save("model_Nvidia.h5")

print(hist_obj.history.keys())

import os

if False: #'DISPLAY' in os.environ:
    import matplotlib.pyplot as plt

    plt.plot(hist_obj.history['loss'])
    plt.plot(hist_obj.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

