# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showing the video result.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes with dropout layers and fully connected layers as well. (model.py lines 71-85)

The model is basically an implementation of NVIDIA's model given in their [research paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) . 
The model includes RELU layers to introduce nonlinearity with Exponential Linear Unit (code 71-85), and the data is normalized in the model using a Keras lambda layer (code line 72).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py lines 79). 

The model was trained and validated on the the dataset provided by Udacity. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86).

#### 4. Appropriate training data

For the training data I ended up using the dataset provided to me by Udacity. In addition to the center, left and right images from the simulator I used the flipped version of these images to increase the training data. A correction factor of +0.2 was applied for the left steering measurement and a correction factor of -0.2 was applied for the right steering measurement.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was described below:

My first step was to use a convolution neural network model similar to the the one created by Nvidia. I thought this model might be appropriate because it was applied on real life autonomous driving conditions.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.To combat the overfitting, I modified the model by adding a dropout layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. This was resolved by augmenting the left and right image data, and also the steering angles. This improved my model but there were shakiness when the car was moving. To resolve it, I have adjusted the steering correction value from 0.4 to 0.2 for both left and right.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 71-90) consisted of a convolution neural network with the following layers and layer sizes.

* Image normalization & Cropping
* Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
* Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
* Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
* Dropout : 0.25 probability
* Fully connected: neurons: 1164, activation: RELU
* Fully connected: neurons: 100, activation: RELU
* Fully connected: neurons: 50, activation: RELU
* Fully connected: neurons: 10, activation: RELU
* Fully connected: neurons: 1 (output)
```
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

The training data was obtained from Udacity provided images (/opt/carnd_p3/data) in the workspace. This data was shuffled and used to 
train the model. As the size of the data increased due to data augmentation I switched to using generators.

To augment the data sat, I also flipped images and angles,
- Randomly shuffled the data set.
- Use OpenCV to convert to RGB for drive.py.
- For steering angle associated with three images, I use correction factor for left and right images with correction factor of 0.2: increase the steering angle by 0.2 for left image and for the right one  decrease the steering angle by 0.2.

![](examples/left.jpg) Left FOV<br>
![](examples/center.jpg) Center FOV<br>
![](examples/right.jpg) Right FOV

After this training, the car was driving down the road all the time on the [first track](https://github.com/thayumaanavan/CarND-Behavioral-Cloning-P3/blob/master/video.mp4).

#### Output
Output video: https://github.com/thayumaanavan/CarND-Behavioral-Cloning-P3/blob/master/video.mp4

(or)

Youtube link : https://youtu.be/4cvz0E5S3HQ
