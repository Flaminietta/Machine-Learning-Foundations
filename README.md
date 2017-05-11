# Machine Learning Foundations

In this repository I collect all the completed assignements for the course "Machine Learning" given by Andrew NG through Coursera.
All the functions have been written in Matlab by myself. I followed the instructions given for each exercise and in the learning sections.

You can find here examples/functions about:

1. LINEAR REGRESSION
- plotData.m - Function to display the dataset
- computeCost.m - Function to compute the cost of linear regression 
- gradientDescent.m - Function to run gradient descent
- computeCostMulti.m - Cost function for multiple variables
- gradientDescentMulti.m - Gradient descent for multiple variables
- featureNormalize.m - Function to normalize features
- normalEqn.m - Function to compute the normal equation

2. LOGISTIC REGRESSION
- plotData.m - Function to plot 2D classification data
- sigmoid.m - Sigmoid Function
- costFunction.m - Logistic Regression Cost Function
- predict.m - Logistic Regression Prediction Function
- costFunctionReg.m - Regularized Logistic Regression Cost

3. MULTICLASS LOGISTIC REGRESSION & NEURAL NETWORK

This exercise is an implementation of one-vs-all logistic regression and neural networks to recognize hand-written digits.

One VS all - multiclass classification

- displayData.m - display 2D a number of digits represented by 20x20 pixels in a grayscale
- lrCostFunction.m - Compute cost and gradient for logistic regression with regularization
- oneVSall.m trains multiple logistic regression classifiers and returns all the classifiers in a matrix (i.e. the trained parameters    
  theta). fmincg.m is called for minimize the cost and find optimal theta.
- predictOneVsAll.m - predict the labels for a trained one-vs-all classifier

Neural network

(The parameters theta are already trained in this exercise)

- predict.m - predict the label of an input given a trained neural network

4. NEURAL NETWORK LEARNING
- displayData.m - Function to help visualize the dataset
- fmincg.m - Function minimization routine (similar to fminunc) 
- sigmoid.m - Sigmoid function
- computeNumericalGradient.m - Numerically compute gradients 
- checkNNGradients.m - Function to help check your gradients
- debugInitializeWeights.m - Function for initializing weights 
- predict.m - Neural network prediction function
- sigmoidGradient.m - Compute the gradient of the sigmoid function 
- randInitializeWeights.m - Randomly initialize weights
- nnCostFunction.m - Neural network cost function

This exercise is an implementation of a neural network model to recognize hand-written digits and goes through the whole process of training a nn:

- Pick a n architecture
- Random initialization of parameters/weights
- Implementation of the Forward Propagation
- Implementation of the cost function
- Implementation of the Back Propagation
- Gradient Checking
- Minimization of the cost function
- Prediction and accuracy



