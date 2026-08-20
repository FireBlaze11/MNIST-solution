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
