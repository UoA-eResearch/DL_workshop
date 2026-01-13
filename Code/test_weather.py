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

seed=1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def create_nn():
    # Input layer
    inputs = keras.Input(shape=(X_data.shape[1],), name='input')

    # Dense layers
    layers_dense = keras.layers.Dense(100, 'relu')(inputs)
    layers_dense = keras.layers.Dense(50, 'relu')(layers_dense)

    # Output layer
    outputs = keras.layers.Dense(1)(layers_dense)

    return keras.Model(inputs=inputs, outputs=outputs, name="weather_prediction_model")

def create_nn_bn():
    # Input layer
    inputs = keras.layers.Input(shape=(X_data.shape[1],), name='input')

    # Dense layers
    layers_dense = keras.layers.BatchNormalization()(inputs)
    layers_dense = keras.layers.Dense(100, 'relu')(layers_dense)
    layers_dense = keras.layers.Dense(50, 'relu')(layers_dense)

    # Output layer
    outputs = keras.layers.Dense(1)(layers_dense)

    # Defining the model and compiling it
    return keras.Model(inputs=inputs, outputs=outputs, name="model_batchnorm")

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=[keras.metrics.RootMeanSquaredError()])

def plot_history(metrics):
    """
    Plot the training history

    Args:
        metrics(str, list): Metric or a list of metrics to plot
    """
    history_df = pd.DataFrame.from_dict(history.history)
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("RMSE")
    plt.show()

def plot_predictions(y_pred, y_true, title):
    plt.style.use('ggplot')  # optional, that's only to define a visual style
    plt.scatter(y_pred, y_true, s=10, alpha=0.5)
    plt.xlabel("predicted sunshine hours")
    plt.ylabel("true sunshine hours")
    plt.title(title)
    plt.show()

filename_data = "weather_prediction_dataset_light.csv"
data = pd.read_csv(filename_data)

#or data = pd.read_csv("https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1")
#data.head()

nr_rows = 365*3
# data
X_data = data.loc[:nr_rows].drop(columns=['DATE', 'MONTH'])

# labels (sunshine hours the next day)
y_data = data.loc[1:(nr_rows + 1)]["BASEL_sunshine"]
print(X_data.shape[1])

X_train, X_holdout, y_train, y_holdout = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_holdout, y_holdout, test_size=0.5, random_state=0)

# ### 
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.tree import plot_tree

# # Define our model
# # extra parameter called n_estimators which is number of trees in the forest
# # a leaf is a class label at the end of the decision tree
# forest = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_leaf=1) 

# # train our model
# forest.fit(X_train, y_train)
# forest_preds = forest.predict(X_test)

# rmse_forest = mean_squared_error(y_test, forest_preds, squared=False)
# print("Forest RMSE: ", rmse_forest)

model = create_nn()
model.summary()
compile_model(model)

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    verbose=2)

plot_history('root_mean_squared_error')

#==========================================================================================
y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)

plot_predictions(y_train_predicted, y_train, title='Predictions on the training set')
plot_predictions(y_test_predicted, y_test, title='Predictions on the test set')

train_metrics = model.evaluate(X_train, y_train, return_dict=True)
test_metrics = model.evaluate(X_test, y_test, return_dict=True)
print(test_metrics)
print('Train RMSE: {:.2f}, Test RMSE: {:.2f}'.format(train_metrics['root_mean_squared_error'], test_metrics['root_mean_squared_error']))

#Tune
y_baseline_prediction = X_test['BASEL_sunshine']
plot_predictions(y_baseline_prediction, y_test, title='Baseline predictions on the test set')

rmse_baseline = mean_squared_error(y_test, y_baseline_prediction, squared=False)
print('Baseline:', rmse_baseline)
print('Neural network: ', test_metrics['root_mean_squared_error'])

model = create_nn()
compile_model(model)

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    validation_data=(X_val, y_val))

plot_history(['root_mean_squared_error', 'val_root_mean_squared_error'])

#Early stopping
model = create_nn()
compile_model(model)

earlystopper = EarlyStopping(
    monitor='val_loss',
    patience=10
    )

history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 200,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper])

plot_history(['root_mean_squared_error', 'val_root_mean_squared_error'])


#Batch normalization
model = create_nn_bn()
compile_model(model)
model.summary()

history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 1000,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper])

plot_history(['root_mean_squared_error', 'val_root_mean_squared_error'])

y_test_predicted = model.predict(X_test)
plot_predictions(y_test_predicted, y_test, title='Predictions on the test set')

rmse_dl = mean_squared_error(y_test, y_test_predicted, squared=False)
print("DL RMSE: ", rmse_dl)

#tensorboard
from keras.callbacks import TensorBoard
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # You can adjust this to add a more meaningful model name
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 200,
                    validation_data=(X_val, y_val),
                    callbacks=[tensorboard_callback],
                    verbose = 2)

#save/load
model.save('my_tuned_weather_model')







