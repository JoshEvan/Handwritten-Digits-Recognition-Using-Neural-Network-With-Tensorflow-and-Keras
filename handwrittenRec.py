import tensorflow as tf
import  cv2
from matplotlib import pyplot as plt
import numpy as np

"""
defining customCallback class
this will allow us to stop the training process when our model reached certain accuracy
"""
class customCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>=0.99):
      print("\n 99% accuracy reached terminating training process")
      self.model.stop_training = True

#####################################################################################################################
# LOAD THE DATASET
mnist = tf.keras.datasets.mnist

"""
THE MNIST DATABASE of handwritten digits consists of 60,000 training grayscale images and 10,000 test grayscale images
the test images are used to test the model with images that were never seen before by the model
the x denotes the image, the input of the model
the y denotes the label or the correct answer, the expected output of the model
our model will learn how to calculate or categorize the input to become the output
"""
(x_train, y_train),(x_test, y_test) = mnist.load_data()

"""
we first normalize each pixel value of our images
so that the prediction score can fall between 0 to 1
"""
x_train = x_train/255.0
x_test = x_test / 255.0
# make an instance of customCallback
callbacks = customCallback()

#####################################################################################################################
# DEFINE THE PREDICTION MODEL
"""
here we define a model, we make a plan to make a model with 3 layers
the first layer : the input layer
  -  it recieves the input image in with size of 28x28 pixels
the second layer : the hidden layer
  -  processing the images with 128 neurons
  -  the activation function at each neuron define the firing rate of the particular neuron
  -  the activation function tells the model what to do next
        -> using the relu function that simply said
            if x > 0:
                return x
            else 
                return 0
            the function will only pass the value that >= 0

the third layer : the output layer
  -  consist of 10 neuron which correspond to each 10 classes of 10 digits (0 to 9)
  -  the model's prediciton answer is one of the 10 neuron that has the highest score
  -  using softmax activation function that simply said take the biggest value only, the unchosen one the value will be changed to 0
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

#####################################################################################################################
# BUILDING THE PREDICTION MODEL
'''
The model is built with sparse_categorical_crossentropy loss function and adam optimizer.
Loss function is used to tell us how good our current model is.
The Optimizer then generate a new guess of model if the Loss function tell us that this current model is not good enough and still need improvement.
The sparse_categorical_crossentropy loss functionis usually used when our output classes are mutually exclusive, such as in this dataset.
'''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#####################################################################################################################
# TRAIN THE PREDICITON MODEL
"""
training the model with 70,000 training images
a supervised learning, where we put the image and also the label which denotes the answer of the current image
"""
with tf.device('/device:GPU:0'):
    model.fit(x_train,y_train,epochs = 10,callbacks=[callbacks])
    """
    @param1 x_train - training images
    @param2 y_train - training labels (correct answer)
    @param3 epochs - how many iteration will the training process do
    @param4 callbacks - alowwing training process to be stopped when reached certain desired amount of accuracy 
                        even when haven't reached the maximum number of iteration
    
    """

#####################################################################################################################
# TEST THE PREDICTION MODEL
"""
the x_test is a list containing all test images of MNIST dataset
"""
images = np.vstack([x_test])
results = model.predict(x_test)

print(model.evaluate(x_test,y_test))

# show the first 10 prediction result
idx = 0
for pred,answer,image in zip(results,y_test,x_test):
    if(idx+1>10):
        # there are 10,000 test images in MNIST dataset
        # we just want to test the first 10 test images
        break
    

    # the pred variable is the model's prediciton result for particular image in the form of an numpy array
    # there are 10 confidence amount that correspond to 10 different digit classes(digit 0 to 9)
    print("Each class confidence/score for image {}: ".format(chr(idx+ord('A'))))
    # print(pred)
    for label,score in enumerate(pred):
        print(str(label)," : ",str(score))
    
    # we pick the class that has the highest confidence score that denotes the model's prefered answer
    text = str(chr(idx+ord('A'))) + ". model's prediction: "+str(pred.argmax()) +"\nthe truth: "+str(answer)
    # show the 10 test images
    plt.subplot(2,5,idx+1)
    plt.title(text)
    plt.imshow(image)
    idx+=1

plt.show()