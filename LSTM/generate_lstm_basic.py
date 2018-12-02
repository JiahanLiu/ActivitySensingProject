import numpy as np
from project_share import *
from tensorflow import keras
from keras.layers import Dense
from keras.layers import LSTM

# architect model
lstm_model = keras.Sequential()
lstm_model.add(LSTM(4, input_shape(), kernel_initializer='normal', activation='relu'))
lstm_model.add(Dense(51, kernel_initializer='normal', activation='sigmoid'))
# Compile model
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

bacc_metric = Project_Metrics(mtst)
history = lstm_model.fit(xtrn,ytrn, epochs=100, batch_size=10000, validation_data=(xtst,ytst), callbacks=[bacc_metric])
print(bacc_metric.get_data())

model.save('lstm_basic.h5')  # creates a HDF5 file 'lstm_basic.h5'

