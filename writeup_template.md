# Traffic Sign Recognition Project Writeup

by Evan Welch

---

## Background

The goals of this project were:
* Load the German Traffic Sign data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a CNN model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./VIZ/examples.png "Examples"
[image2]: ./VIZ/distribution.png "Distribution"
[image3]: ./VIZ/before_after.png "Before & After"
[image4]: ./DATA/online_examples/51786163-children-crossing-German-road-sign-Stock-Photo.jpg "Online 1"
[image5]: ./DATA/online_examples/mifuUb0.jpg "Online 2"
[image6]: ./DATA/online_examples/64914157-German-road-sign-slippery-road-Stock-Photo.jpg "Online 3"
[image7]: ./DATA/online_examples/35510405-German-sign-warning-about-wild-animals-like-deer-crossing-the-road--Stock-Photo.jpg "Online 4"
[image8]: ./DATA/online_examples/5155701-German-traffic-sign-No-205-give-way-Stock-Photo.jpg "Online 5"

-----

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Files Submitted

###### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project notebook](https://github.com/evanfwelch/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

###### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used basic numpy operations to calculate summary statistics of the traffic signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43


####2. Include an exploratory visualization of the dataset.

Here is a graph of some example images from the data and their labels.

![Example images][image1]

And here is a graph that shows the distribution of classes (`n_classes = 43`) across all training, validation, and test data:

![Class distribution][image2]


### Design and Test a Model Architecture

###### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I preprocessed the images in two steps: converting to grayscale and normalizing them from `[0,256]` to `[-1,1]`.

My guiding principle in this project *do the simplest possible thing, and see if I can get it to work*. In this vein, I decided to use grayscale so I could have fewer degrees of freedom when hand-tweaking the architecture. Moreover, I intuitively felt like there was enough geometrically distinct about the different signs to warrant ignoring colors like blue and red. Given more time I'd like to more carefully test the efficacy of including color.

As for normalization, I wanted to follow best practices by having my weights be initialized to random, small values normally distributed around zero. It makes sense to have my pixel data be on the same order of magnitude as my weights themselves.

Here is an example of a traffic sign image before and after Pre-processingP

![preprocesing before and after][image3]

If the architecture were struggling to achieve good validation accuracy, I would have generated additional data with random shifts and rotations. But I figured there's no need to complicate the pipeline when a simple approach is remarkably accuracte.



###### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  VALID padding, outputs 14x14x64 	|
| Convolution 5x5	    | 1x1 stride, VALID padding, output 10x10x16    									|
| RELU |    |												|
| Max pooling	      	| 2x2 stride,  VALID padding, outputs 5x5x16	|
| Flatten | outputs 400x1 |
| Fully connected	 |	 400x200, outputs 200x1   									|
| RELU | |
| Dropout | 50% dropout in training|
| Fully connected | 200x100, outputs 100x1 |
| RELU |   |
| Dropout | 50% dropout in training |
| Fully connected | 100 x 43, outputs logits [43x1] |


###### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
* An `AdamOptimzer` minimizing average cross entryp, with a learning rate of :
* `learning_rate = 0.001`

It ran for `EPOCHS=15`, with `BATCH_SIZE=64`.

###### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995,
* validation set accuracy of 0.947, and
* test set accuracy of 0.930

Arguably there is some overfitting here because the training accuracy is so high relative to the validation set. But

I started with the LeNet architecture and did the minimum modifications of the first layer to make it compatible with 32x32 grayscale images. This was more a choice of convenience and familiarity rather than anything model specific. That being said, I think it's generally good practice to start with a simple, well-known benchmark and make changes only as needed.

LeNet was geared for 28x28 images and, because it was classifying among 10 distinct options instead of 43, I judged that the fully connected layers "tapered" down too quickly. Moreover, perhaps because the MNIST images were more consistent and centered appropriately, the LeNet architecture wasn't generalizing well to this traffic sign dataset. I also was experiencing some overfitting. To combat this is I introduced dropout before each fully connected layer. Dropout is a very intuitive and elegant way to reduce the specific dependencies on any one parameter. This helped a great deal.

I reduced the learning rate from 0.005 to 0.001 because I had scarcity of training time and thus did not need to learn especially quickly. I also reduced the batch size to 64 and increased the number of epochs to 15, both because I wasn't seeing any overfitting with the LeNet parameters. As long as you're not overfitting and the training time is reasonable, one might as well fit the data a little more aggressively.

All of these choices were done "By hand". In production I full well would expect to choose all these hyperparameters with cross-validation. For the purposes of this project, I felt obsessive optimization would have diminishing "educational returns".


### Test a Model on New Images

###### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

While all of these images appear are high resolution JPEGs that aren't blurry or obscured, I used `scipy.misc.imread` and `scipy.misc.imresize` to read and downsample them to 32x32x3 images. I did not bother to do any cropping or centering which could prove to be a challenge for classification.

###### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Children Crossing      		| Traffic signals   									|
| Road Work     			| Wild Animals Crossing										|
| Slippery Road					| Slippery Road											|
| Wild Animals Crossing      		|  Wild Animals Crossing					 				|
| Yield			| Speed Limit (80 km/h)      							|

Unfortunately the model was only able to correctly predict 2 out of the 5 "out of sample" images. This compares to a test accuracy of 93%. This misclassification is a bit puzzling because the resized and pre-processed images *I found* don't look too much grainier or blurrier than the typical images that appear in the training data. With more time, I would try cropping the out of sample data to see if that resolved the error. If it didn't, I would try training the classifier on "fake" data in hopes of developing a more generalizable classifier.

###### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


Image 1: Children crossing

| Description | Softmax prob |
| ------- | ------- |
|Roundabout mandatory |  10.2% |
|Go straight or left |  9.9% |
|Speed limit (30km/h) |  8.7% |
|Road narrows on the right |  7.0% |
|Speed limit (20km/h) |  6.6% |

Image 2: Road work

| Description | Softmax prob |
| ------- | ------- |
|Road work |  99.8% |
|Beware of ice/snow |  0.2% |
|Double curve |  0.0% |
|Bumpy road |  0.0% |
|Bicycles crossing |  0.0% |

Image 3: Slippery road

| Description | Softmax prob |
| ------- | ------- |
|Slippery road |  100.0% |
|Dangerous curve to the left |  0.0% |
|Dangerous curve to the right |  0.0% |
|Wild animals crossing |  0.0% |
|Beware of ice/snow |  0.0% |

Image 4: Wild animals crossing

| Description | Softmax prob |
| ------- | ------- |
|Wild animals crossing |  100.0% |
|Road work |  0.0% |
|Speed limit (80km/h) |  0.0% |
|Double curve |  0.0% |
|No passing for vehicles over 3.5 metric tons |  0.0% |

Image 5: Yield

| Description | Softmax prob |
| ------- | ------- |
|Yield |  100.0% |
|Priority road |  0.0% |
|Turn right ahead |  0.0% |
|Ahead only |  0.0% |
|Keep left |  0.0% |
