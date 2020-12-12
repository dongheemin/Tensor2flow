#Tensorflow Tutorial Image Classfication

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pltgraph as pg

#1. Version Check
#tf.__version__
#Output ==> '2.3.1'

#2. Load Dataset by Fasion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shrit/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shrit', 'Sneaker',
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
for i in range (0, 10):
	plt.figure(figsize=(5,5))
	plt.imshow(train_images[i])
	plt.colorbar()
	plt.grid(False)
	plt.savefig('./images/fig'+str(i)+'.png', dpi=150)

# Noramlization Dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

# Visualization Normalization Dataset[0-24]
plt.figure(figsize=(5,5))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	plt.xlabel(class_names[train_labels[i]])
plt.savefig('./images/NorFig.png', dpi=150)

# Model Making
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation='relu'),
	keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# Model evaluation

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\n데이터 정확도:', test_acc)

# Make Prediction
predictions = model.predict(test_images)

print('예측값 :',np.argmax(predictions[0]))
print('실제값 :',test_labels[0])

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt = pg.plot_image(plt, i, predictions, test_labels, test_images, class_names)
plt.subplot(1,2,2)
plt = pg.plot_value_array(plt, i, predictions, test_labels)
plt.savefig('./images/PredFig.png', dpi=150)
