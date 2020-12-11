#Tensorflow Tutorial Image Classfication

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#1. Version Check
#tf.__version__
#Output ==> '2.3.1'

#2. Load Dataset by Fasion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_name = ['T-shrit/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shrit', 'Sneaker',
		 'Bag', 'Ankle boot']

#2-1. Label - Class
# T-shirt/Top     0
# Trouser         1
# Pullover        2
# Dress           3
# Coat            4
# Sandal          5
# Shirt           6
# Sneaker         7
# Bag             8
# Ankle boot      9

#Check Train Dataset
#train_image.shape
#len(train_labels)
#train_labels

# Output ==> 	(60000, 28, 28)
#		60000
#		array([9,0,0,... 3,0,5], dtype=unit8)

#Check Test Dataset
#test_image.shape
#len(test_labels)
test_labels

# Output ==> 	(10000, 28, 28)
#		10000
#		test

# Visualization Dataset[0-9]
try:
	for i in range (0, 10):
		plt.figure(figsize=(28,28))
		plt.imshow(train_images[i])
		plt.colorbar()
		plt.grid(False)
		plt.savefig('./images/fig'+str(i)+'.png', dpi=300)
except Exception as e :
	print(e)

