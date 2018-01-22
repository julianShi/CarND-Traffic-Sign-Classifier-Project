# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image20]: ./examples/color_image.jpg "Color image"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./benchmark_ini/07.png "Traffic Sign 1"
[image5]: ./benchmark_ini/13.png "Traffic Sign 2"
[image6]: ./benchmark_ini/14.png "Traffic Sign 3"
[image7]: ./benchmark_ini/18.png "Traffic Sign 4"
[image8]: ./benchmark_ini/35.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/julianshi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many instances of each class are there. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to take only one channel of the three-channel png images. Because the gray, shape, edge, position and orientation information of images have been complete in one channel. The rest two channels add much complexity to the neural network, yet contribute little to the information. 

Here is an example of a traffic sign image with three colors channels and red channel.

![alt text][image20]
![alt text][image2]

As a last step, I normalized the image data because the normalized input data value helps to boost the efficiency of training. 

I decided not to generate additional data because image editing technics like the edge detection can be achieved by the hidden neural network layers. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 28x28x6	 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 28x28x6	 				|
| Flatten				|												|
| Fully connected		| Input = 5x5x16. Output = 400.        			|
| RELU					|												|
| Fully connected		| Input = 400. Output = 120.        			|
| RELU					|												|
| Fully connected		| Input = 120. Output = 84.        			|
| RELU					|												|
| Fully connected		| Input = 84. Output = 43. 		       			|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an optimizer where batch size is 100, number of epochs is 50, learning rate is 0.001.
I compared different batch sizes, including 100, 200, 500, 1000, where 100 shot the highest accuracy after one epoch. 
Displayed on the ipynb file is the last 20 epochs when the accuracy improved very slow. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.888
* test set accuracy of 0.878


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]
![alt text][image5| width=128]
![alt text][image6| width=128] 
![alt text][image7| width=128] 
![alt text][image8| width=128]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100 km/h      		| No vehicles   									| 
| Yield					| Yield											|
|  Priority road     			| Priority road 										|
| General caution	    | General caution					 				|
| Ahead only			| Ahead only      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model wrongly predicted the 100 km/h sign to be "no vehicles" (probability of 0.69), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.696 | No vehicles |
| 0.304 | Speed limit (80km/h) |
| 0.000 | Speed limit (50km/h) |
| 0.000 | Speed limit (60km/h) |
| 0.000 | Ahead only |

For the rest, the prediction had been quite accurate. 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 | Yield |
| 0.000 | Speed limit (20km/h) |
| 0.000 | Speed limit (30km/h) |
| 0.000 | Speed limit (50km/h) |
| 0.000 | Speed limit (60km/h) |

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 | Stop |
| 0.000 | Speed limit (30km/h) |
| 0.000 | Roundabout mandatory |
| 0.000 | Speed limit (20km/h) |
| 0.000 | Speed limit (50km/h) |

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 | General caution |
| 0.000 | Speed limit (20km/h) |
| 0.000 | Speed limit (30km/h) |
| 0.000 | Speed limit (50km/h) |
| 0.000 | Speed limit (60km/h) |

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 | Ahead only |
| 0.000 | Speed limit (20km/h) |
| 0.000 | Speed limit (30km/h) |
| 0.000 | Speed limit (50km/h) |
| 0.000 | Speed limit (60km/h) |








