# PythonBackendDigitRecognition

This is the backend python code for the https://github.com/AndrewBarzu/angular-digit-recognition app.
All it does is get an image in the form of a get request, and feed forward it through a Fully Connected Neural Network.
After it does that, it returns the predicted result back to the sender.

## Neural Network

The Neural Network was pretrained, and all i kept from the training were the weights resulted (in Theta1.mat and Theta2.mat for forward propagation from input layer
to hidden layer and hidden layer to output layer respectively).

As it can be inferred, the neural network uses 3 fully connected layers (only 1 hidden layer, with 16 units). 
