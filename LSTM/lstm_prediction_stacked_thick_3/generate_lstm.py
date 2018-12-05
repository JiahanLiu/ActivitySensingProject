import numpy as np
from project_share import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pickle

# architect model
lstm_model = Sequential()
lstm_model.add(LSTM(60, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer='normal', activation='relu'))
lstm_model.add(LSTM(60, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer='normal', activation='relu'))
lstm_model.add(LSTM(60, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer='normal', activation='relu'))
lstm_model.add(Dense(51, kernel_initializer='normal', activation='sigmoid'))
# Compile model
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#need to adjust for missing rows
mtst_timeseries = mtst[1:]; 
bacc_metric = Project_Metrics(mtst_timeseries)
history = lstm_model.fit(train_X, train_Y, epochs=100, batch_size=10000, validation_data=(test_X, test_Y), callbacks=[bacc_metric])
print(bacc_metric.get_data())

lstm_model.save('lstm_step1_prevonly_lstm1.h5')  # creates a HDF5 file 'lstm_basic.h5'
with open('bacc_metric_simple.pkl', 'wb') as output:  # Overwrites any existing file.
    pickle.dump(bacc_metric.get_data(), output)
with open('bacc_metric_detailed.pkl', 'wb') as output:  # Overwrites any existing file.
    pickle.dump(bacc_metric.get_detailed_data(), output)