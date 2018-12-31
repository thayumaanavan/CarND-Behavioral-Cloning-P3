import csv
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from scipy import ndimage
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Cropping2D, Conv2D, MaxPooling2D


#load data
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skips header line
    for line in reader:
        samples.append(line)

# Read images from the training
def read_image(path):
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # drive.py loads images in rgb
    return image_rgb

def generator_images(samples, batch_size = 32):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            steering_correction = 0.2
            # For each line in the driving data log, read camera image (left, right and centre) and steering value
            for batch_sample in batch_samples:
                image_centre = read_image('data/'+ '/IMG/' + line[0].split('/')[-1])
                image_left = read_image('data/'+ '/IMG/' + line[1].split('/')[-1])
                image_right = read_image('data/'+ '/IMG/' + line[2].split('/')[-1])
                steering_centre = float(batch_sample[3])
                steering_left = steering_centre + steering_correction
                steering_right = steering_centre - steering_correction
                images.extend([image_centre, image_left, image_right])
                measurements.extend([steering_centre, steering_left, steering_right])
            
            # Augment training data by flipping images and changing sign of steering
            augmented_images, augmented_measurements = [], []
            for image,measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0) 
                
            X_train = np.array(augmented_images)
            Y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, Y_train)

# split driving data to train and validate
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Use generator to pull data 
batch_size=32
train_generator = generator_images(train_samples, batch_size)
validation_generator = generator_images(validation_samples, batch_size)

#NVIDIA Architecture
model = Sequential()
model.add(Lambda(lambda x : x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Conv2D(filters= 24, kernel_size = 5, strides = 2, activation='relu'))
model.add(Conv2D(filters= 36, kernel_size = 5, strides = 2, activation='relu'))
model.add(Conv2D(filters= 48, kernel_size = 5, strides = 2, activation='relu'))
model.add(Conv2D(filters= 64, kernel_size = 3, strides = 1, activation='relu'))
model.add(Conv2D(filters= 64, kernel_size = 3, strides = 1, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print(model.summary())

model.compile(loss = 'mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch = int(len(train_samples)/batch_size),
                    validation_data = validation_generator, nb_val_samples = int(len(validation_samples)/batch_size),
                    nb_epoch = 3)
model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


        