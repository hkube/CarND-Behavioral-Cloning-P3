#!python

import csv
import matplotlib.image as mpimg
import numpy as np
import os
from numpy import imag

DATA_DIR='./data1'

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
    orig_img_path = line[0]
    local_img_path = os.path.join(DATA_DIR, "IMG", os.path.basename(orig_img_path))
    image = mpimg.imread(local_img_path)
    if img_shape is None:
        img_shape = image.shape
    images.append(image)
    angle = float(line[3])
    steering_angles.append(angle)
    #print("path:", local_img_path, "  shape:", image.shape, "  steering angle:", angle)
    #exit()

X_train = np.array(images)
y_train = np.array(steering_angles)



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=img_shape))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, validation_split=0.2, shuffle="True", nb_epoch=3, verbose=2)
model.save("model.h5")
