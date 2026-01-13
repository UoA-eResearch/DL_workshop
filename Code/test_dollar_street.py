import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from tensorflow import keras

import seaborn as sns
import pandas as pd

seed=1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# train_images = np.load("./dollar_street_10/train_images.npy")
# train_labels = np.load("./dollar_street_10/train_labels.npy")
# test_images = np.load("./dollar_street_10/test_images.npy")
# test_labels = np.load("./dollar_street_10/test_labels.npy")

train_images = np.load('/content/train_images.npy')
train_labels = np.load('/content/train_labels.npy')
test_images = np.load('/content/test_images.npy')
test_labels = np.load('/content/test_labels.npy')

print(train_images.shape)
print(f'Min val: {train_images.min()}, Max val: {train_images.max()}')

print(train_labels.shape)
print(f'Min val: {train_labels.min()}, Max val: {train_labels.max()}')

train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['day bed', 'dishrag', 'plate', 'running shoe', 'soap dispenser',
                'street sign', 'table lamp', 'tile roof', 'toilet seat', 'washing machine']

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.axis('off')
    plt.title(class_names[train_labels[i]])
plt.show()

dim = train_images.shape[1] * train_images.shape[2] * train_images.shape[3]
print(f'Number of features: {dim}')

#Define model
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="model_small")
model.summary()

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df[['accuracy', 'val_accuracy']])
plt.show()
sns.lineplot(data=history_df[['loss', 'val_loss']])
plt.show()

model.save('model_small')

# Dropout
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.Dropout(0.8)(x) # This is new!
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)

model_dropout = keras.Model(inputs=inputs, outputs=outputs, name="model_dropout")

model_dropout.summary()
model_dropout.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history_dropout = model_dropout.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

history_df = pd.DataFrame.from_dict(history_dropout.history)
history_df['epoch'] = range(1,len(history_df)+1)
history_df = history_df.set_index('epoch')
sns.lineplot(data=history_df[['accuracy', 'val_accuracy']])
plt.show()

test_loss, test_acc = model_dropout.evaluate(test_images,  test_labels, verbose=2)
print(f'Test acc: {test_acc * 100} %')
sns.lineplot(data=history_df[['loss', 'val_loss']])
plt.show()

# DenseNet121 prebuilt model - Transfer learning 
# input tensor
inputs = keras.Input(train_images.shape[1:])

# upscale layer
method = tf.image.ResizeMethod.BILINEAR
upscale = keras.layers.Lambda(
  lambda x: tf.image.resize_with_pad(x, 160, 160, method=method))(inputs)

base_model = keras.applications.DenseNet121(include_top=False,
                                            pooling='max',
                                            weights='imagenet',
                                            input_tensor=upscale,
                                            input_shape=(160,160,3),
                                            )

base_model.trainable = False

out = base_model.output
out = keras.layers.Flatten()(out)
out = keras.layers.BatchNormalization()(out)
out = keras.layers.Dense(50, activation='relu')(out)
out = keras.layers.Dropout(0.5)(out)
out = keras.layers.Dense(10)(out)

model = keras.models.Model(inputs=inputs, outputs=out, name = 'tfl_model')

history_tfl = model_dropout.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

history_df = pd.DataFrame.from_dict(history_tfl.history)
history_df['epoch'] = range(1,len(history_df)+1)
history_df = history_df.set_index('epoch')
sns.lineplot(data=history_df[['accuracy', 'val_accuracy']])
plt.show()

test_loss, test_acc = model_dropout.evaluate(test_images,  test_labels, verbose=2)
print(f'Test acc: {test_acc * 100} %')
sns.lineplot(data=history_df[['loss', 'val_loss']])
plt.show()

