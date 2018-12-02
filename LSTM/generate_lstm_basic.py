import numpy as np
from project_share import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# architect model
lstm_model = Sequential()
lstm_model.add(LSTM(60, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer='normal', activation='relu'))
lstm_model.add(Dense(51, kernel_initializer='normal', activation='sigmoid'))
# Compile model
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

bacc_metric = Project_Metrics(mtst)
history = lstm_model.fit(train_X, train_Y, epochs=100, batch_size=10000, validation_data=(test_X, test_Y), callbacks=[bacc_metric])
print(bacc_metric.get_data())

model.save('lstm_basic.h5')  # creates a HDF5 file 'lstm_basic.h5'
