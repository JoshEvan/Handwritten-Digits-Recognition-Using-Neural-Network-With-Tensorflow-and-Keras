# Handwritten Digits Recognition Using Neural Network With Tensorflow and Keras

This code will give your computer the ability to recognize handwritten digits.

## How It Works
This code is using the TensorFlow library with [Keras](https://www.tensorflow.org/beta/guide/keras), a high-level API for deep learning.

The code is divided into 5 sections:
1. #### Section 1
   Load and preprocess the datasets and define a callback class that will be used on the training process. 
2. #### Section 2
   Define the prediction model.

   The Neural Network model is defined as follows:

   The Neural Network consists of 3 layers

   i. The first layer : the input layer

     - it recieves the input image in with size of 28x28 pixels.

   ii. The second layer : the hidden layer
      -  processing the images with 128 neurons
      -  the activation function at each neuron define the firing rate of the particular neuron
      -  the activation function tells the model what to do next
        - using the relu function that simply said
        
        ```
            if x > 0:
                return x
            else 
                return 0

            the function will only pass the value that >= 0    
        ```
     iii. The third layer : the output layer
      -  consist of 10 neurons which corresponds to every 10 classes of 10 digits (0 to 9)
      -  the model's prediction answer is one of the 10 neurons that has the highest score
      -  using softmax activation function that simply said take the biggest value only, the unchosen one the value will be changed to 0

3. #### Section 3
   Build the Prediction Model
   The model is built with sparse_categorical_crossentropy loss function and adam optimizer.
   
   Loss function is used to tell us how good our current model is.
   
   The Optimizer then generate a new guess of model if the Loss function tell us that this current model is not good enough and still need improvement.
   
   The sparse_categorical_crossentropy loss functionis usually used when our output classes are mutually exclusive, such as in this dataset.
   
5. #### Section 4
   Train the Prediciton Model
   Training the model with 70,000 training images.
   
   A supervised learning, where we put the image and also the label which denotes the answer of the current image.
   
   We train the data within maximum of 10 iterations, which will be automatically stopped when the model reached 99% accuarcy.
7. #### Section 5
   Test the Prediciton Model



## Datasets
We are using THE MNIST DATABASE of handwritten digits that consists of 60,000 training grayscale images and 10,000 test grayscale images. The dataset is currently available on the Keras. The dataset is able to be directly accessed within the code.

## Installation
What things you need to install and how to install them:

Make sure you have Python 3 working on your system. This code was made with Python 3.6.5.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```
pip install numpy
pip install matplotlib
pip install opencv-python
pip install tensorflow
```
or if you are using Anaconda
```
conda install -c anaconda numpy
conda install -c conda-forge matplotlib
conda install -c menpo opencv
conda install -c conda-forge tensorflow
```

## Usage
1. Clone or download the repository
2. Install all the prerequisites
3. Open your Command Prompt or Terminal or Anaconda Prompt
4. Navigate to your current file directory
5. Run following command

```
python handwrittenRec.py
```
## Result
![Result Image](https://raw.githubusercontent.com/JoshEvan/Handwritten-Digits-Recognition-Using-Neural-Network-With-Tensorflow-and-Keras/master/handwrittenRec.PNG)

