import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('mnist_train.csv')

data.head()

data = np.array(data)
rows, columns = data.shape
# columns will be one extra for the "label" column, use print(data.head()) to see this
np.random.shuffle(data)

data_dev = data[0:1000].T # for the first 1000 rows # .T is transposing (flipping rows and columns) this is because its easier to see each example as a row
Y_dev = data_dev[0]
X_dev = data_dev[1:columns] / 255

data_train = data[1000:rows].T # for the rest of the rows
Y_train = data_train[0]
X_train = data_train[1:columns] / 255

def ReLU(x):
    return np.maximum(0, x)
    
def forward_prop(W1, b1, W2, b2, x):
    # matrix multiplication for 2D arrays ONLY
    Z1 = W1.dot(x) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    #dW2 = 1 / rows * dZ2.dot(A1.T)
    #db2 = 1 / rows * np.sum(dZ2, axis=1, keepdims=True)
    dW2 = 1 / X.shape[1] * dZ2.dot(A1.T)
    db2 = 1 / X.shape[1] * np.sum(dZ2, axis = 1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_of_ReLU(Z1)
    #dW1 = 1 / rows * dZ1.dot(X.T)
    #db1 = 1 / rows * np.sum(dZ1, axis=1, keepdims=True)
    dW1 = 1 / X.shape[1] * dZ1.dot(X.T)
    db1 = 1 / X.shape[1] * np.sum(dZ1, axis = 1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):  # this is to tweak the parameters everytime the network loops on itself 
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    print(f"Iterations to loop through: {iterations}")
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 25 == 0):
            print(f"Iteration no: {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(A2), Y)}")
    return W1, b1, W2, b2
