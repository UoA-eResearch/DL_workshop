import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


penguins = sns.load_dataset('penguins')
#print(penguins)

penguins['species'] = penguins['species'].astype('category')
# Drop two columns and the rows that have NaN values in them
penguins_filtered = penguins.drop(columns=['island', 'sex']).dropna()

# Extract columns corresponding to features
penguins_features = penguins_filtered.drop(columns=['species'])
target = pd.get_dummies(penguins_filtered['species'])

X_train, X_test, y_train, y_test = train_test_split(penguins_features, target,test_size=0.2, random_state=0, shuffle=True, stratify=target)

seed=1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

inputs = keras.Input(shape=X_train.shape[1])
hidden_layer = keras.layers.Dense(10000, activation="relu")(inputs)
output_layer = keras.layers.Dense(3, activation="softmax")(hidden_layer)
model = keras.Model(inputs=inputs, outputs=output_layer)
model.summary()

model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs=1000)
scores = model.evaluate(X_test, y_test)
print(scores)

sns.lineplot(x=history.epoch, y=history.history['loss'])
plt.show()
y_pred = model.predict(X_test)
prediction = pd.DataFrame(y_pred, columns=target.columns)

predicted_species = prediction.idxmax(axis="columns")
true_species = y_test.idxmax(axis="columns")

matrix = confusion_matrix(true_species, predicted_species)
print(matrix)

# Convert to a pandas dataframe
confusion_df = pd.DataFrame(matrix, index=y_test.columns.values, columns=y_test.columns.values)

# Set the names of the x and y axis, this helps with the readability of the heatmap.
confusion_df.index.name = 'True Label'
confusion_df.columns.name = 'Predicted Label'
sns.heatmap(confusion_df, annot=True)
plt.show()
#%%
import seaborn as sns
penguins = sns.load_dataset('penguins')
penguins['species']
#%%
