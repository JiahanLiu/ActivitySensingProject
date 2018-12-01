import numpy as np
from project_share import *
from tensorflow import keras

# architect model
mlp_model = keras.Sequential()
mlp_model.add(keras.layers.Dense(60, input_dim=175, kernel_initializer='normal', activation='relu'))
mlp_model.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))
# Compile model
mlp_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

bacc_metric = Project_Metrics(mtst)
history = mlp_model.fit(xtrn,ytrn, epochs=100, batch_size=10000, validation_data=(xtst,ytst), callbacks=[bacc_metric])
print(bacc_metric.get_data())