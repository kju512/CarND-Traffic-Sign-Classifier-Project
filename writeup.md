#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./MywriteupPicture/classhbar.png "Visualization"
[image2]: ./MywriteupPicture/eachclassshow1.png "Visualization"
[image3]: ./MywriteupPicture/eachclassshow2.png "Visualization"
[image4]: ./MywriteupPicture/eachclassshow3.png "Visualization"
[image5]: ./MywriteupPicture/eachclassshow4.png "Visualization"
[image6]: ./MywriteupPicture/eachclassshow5.png "Visualization"
[image7]: ./MywriteupPicture/eachclassshow6.png "Visualization"
[image8]: ./MywriteupPicture/eachclassshow7.png "Visualization"
[image9]: ./MywriteupPicture/eachclassshow8.png "Visualization"
[image10]: ./MywriteupPicture/eachclassshow9.png "Visualization"
[image11]: ./MywriteupPicture/eachclassshow10.png "Visualization"
[image12]: ./MywriteupPicture/eachclassshow11.png "Visualization"
[image13]: ./MywriteupPicture/eachclassshow12.png "Visualization"
[image14]: ./MywriteupPicture/eachclassshow13.png "Visualization"
[image15]: ./MywriteupPicture/eachclassshow14.png "Visualization"
[image16]: ./MywriteupPicture/eachclassshow15.png "Visualization"
[image17]: ./MywriteupPicture/grayscale.png "Grayscaling"
[image18]: ./MywriteupPicture/normalization.png "normalizescaling"
[image19]: ./MywriteupPicture/translation.png "Visualization"
[image20]: ./MywriteupPicture/addnoise.png "Visualization"
[image21]: ./MywriteupPicture/resize.png "Visualization"
[image22]: ./MywriteupPicture/00001.png "Traffic Sign 1"
[image23]: ./MywriteupPicture/00002.png "Traffic Sign 2"
[image24]: ./MywriteupPicture/00003.png "Traffic Sign 3"
[image25]: ./MywriteupPicture/00004.png "Traffic Sign 4"
[image26]: ./MywriteupPicture/00005.png "Traffic Sign 5"
[image27]: ./MywriteupPicture/conv1.png "conv1 featuremap"
## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kju512/CarND-Traffic-Sign-Classifier-Project/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32X32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a horizontal bar chart showing the numbers of each class and their names. this shows that some class has much more data than other classes.

![alt text][image1]

And I choose one image from each class to show how each class of the image looks like.It is as follows:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

![alt text][image16]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At the first step, I decided to convert the images to grayscale because we only concern the shape of the traffic sign but not its color.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image17]

 Then, I normalized the image data because this method can convert the value of every feature of sample into the range between -0.5 and 0.5. This benifits training convengence and it can avoid some feature have more impact than other feature.Because each feature has the similar scale.

![alt text][image18]

I decide to generate additional data because each class dataset are not very balance.some class has more data than others.so I will generate more data to make them balance. 

To add more data to the current data set, I used the following techniques because these can augment the data,and they are derived from the exsiting data.

* translate the image
* add some noise to the image
* resize the image

Here is an example of an original image and an augmented image:

![alt text][image19]
![alt text][image20]
![alt text][image21]
After these augumentation,I make the trainning set reach 86000 and each class has 2000 samples.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grey image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| Max pooling           | 2x2 stride,  outputs 16x16x64                 |
| Convolution 3x3       | 1x1 stride, same padding, outputs 16x16x64    |
| Max pooling           | 2x2 stride,  outputs 8x8x64                   |
| Convolution 3x3       | 1x1 stride, same padding, outputs 8x8x64      |
| Max pooling           | 2x2 stride,  outputs 4x4x64                   |
| flatten               | 4x4x64 inputs,  outputs 1024                  |
| dropout               | keep probability 0.55                         |
| Fully connected       | 1024 inputs,  outputs 64                      |
| dropout               | keep probability 0.55                         |
| Fully connected       | 64 inputs,  outputs 64                        |
| Softmax				| 64 inputs,    outputs 43         		    	|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer to optimize the loss.The learning rate is set to be 0.001.The epoch is set to be 40.In fact,it only spend 25 epoches to train the model reaching the destination (the validation accuracy will be greater than 0.94) and the batch size is 128.
At the beginning of training,I suffled the data,and then feed each batch of data to the session.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

At the beginning of my trainning,I choose the LENET5 as my initial model.but it didn't work very well.Its validation accuracy just can reach 0.81.In order to improve the model ,I adjust its architecture.The main changes are as follows:

* The convolutional layer's padding type is changed from "VALID" to "SAME"
* The convolutional layer's kernel size is changed from 5x5 to 3x3
* Increase one convolutional layer
* Change the node quatites of some layer and add two dropout layer to avoid overfitting

When I try train the model,I fine tuned some hyper parameters such as  epoch num, initial learning rate,the node quatites of fullyconnected layers,dropout keep probility.
At last,I got a good output. The training set accuracy and validation set accuracy are both high.And it also works well on test set.The test set accuracy can reach 0.89483.

My final model results were:

* training set accuracy of 0.99297
* validation set accuracy of 0.94163
* test set accuracy of 0.89483

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image22] ![alt text][image23] ![alt text][image24] 
![alt text][image25] ![alt text][image26]

The last image might be difficult to classify because it is very dark.Even if a human nearly can not recognize what the sign is,so it is a difficult one.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			              |     Prediction	        					| 
|:---------------------------:|:-------------------------------------------:| 
| Speed limit (30km/h)        | Speed limit (30km/h)	  				    | 
| Keep right     		      | Keep right 									|
| Turn right ahead		      | Turn right ahead							|
| Right-of-way at the next intersection|Right-of-way at the next intersection|
| Priority Road			      | Priority Road      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 89.483%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure that this is a Speed limit (30km/h) (probability of 0.991044402), and the image does contain a Speed limit (30km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.991044402         	| Speed limit (30km/h)  						| 
| 0.00424366118    		| Speed limit (50km/h) 							|
| 0.00420565484			| Speed limit (70km/h)							|
| 0.000174031578 		| Speed limit (80km/h)			 				|
| 0.000161014454	    | Speed limit (20km/h)    						|
 

For the second image, the top five soft max probabilities were 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Keep right                                    | 
| 6.09569835e-14        | Yeild                                         |
| 5.88930827e-14        | Turn left ahead                               |
| 3.02900549e-15        | Road work                                     |
| 9.57539407e-16        | Go straight or right                          |

For the third image, the top five soft max probabilities were 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 9.72435534e-01        | Turn right ahead                              | 
| 1.39123276e-02        | Go straight or right                          |
| 5.17773302e-03        | No entry                                      |
| 4.71990369e-03        | Stop                                          |
| 1.28055317e-03        | Ahead only                                    |


For the forth image, the top five soft max probabilities were 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 9.99995112e-01        | Right-of-way at the next intersection         | 
| 4.63173274e-06        | Beware of ice snow                            |
| 2.33758954e-07        | Pedestrians                                   |
| 5.67314196e-08        | Road work                                     |
| 1.52712101e-08        | Traffic signals                               |

For the fifth image, the top five soft max probabilities were 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 9.95890498e-01        | Priority Road                                 | 
| 2.38189753e-03        | Roundabout mandatory                          |
| 6.21394720e-04        | Turn right ahead                              |
| 3.37490434e-04        | Ahead only                                    |
| 1.79444993e-04        | Go straight or right                          |
  
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I use the function to Visualizie the conv1 feature map,it is as belows:

![alt text][image27]

It shows that the network use convolutional layer to extract some shapes which is very similar to the raw image. Those shapes can denote what the image is.
