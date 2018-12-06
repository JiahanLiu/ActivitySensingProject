import numpy as np
import pickle
from project_share import *
import tensorflow as tf
from tensorflow import keras

mlp_input_dim = 175
model_dir = "./models/"
metric_dir = "./metrics/"
#create models
mlp_1_32 = keras.Sequential()

mlp_1_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_1_32.add(keras.layers.Reshape((4,8),input_shape=(32,)))
mlp_1_32.add(keras.layers.Conv1D(16, 4, activation='relu', input_shape=(4,8)))
mlp_1_32.add(keras.layers.Flatten())
mlp_1_32.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))


mlp_models = {
"mlp_1_32"              : mlp_1_32,
}

mlp_1_32_metrics            = Project_Metrics(mtst)

mlp_metrics = {
"mlp_1_32"              : mlp_1_32_metrics,
}
"""
# Objective fucntion for multitask, averaging log loss across all labels
# while ignoring missing or Nan values
def multitask_loss(y_true, y_pred):
    
    y_true_cut = []
    y_pred_cut = []
    
    for i in range(len(y_true)):
        if type(y_true[i]) == int:
            y_true_cut.append(y_true[i])
            y_pred_cut.append(y_pred[i])
            
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))
"""
MASK_VALUE = -1

ytst = np.ma.MaskedArray(ytst, mask = mtst)
ytst.fill_value = MASK_VALUE
ytst = ytst.filled()

def build_masked_loss(loss_function, mask_value=MASK_VALUE):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, mask_value), float)
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function

def open_metric(mdl_name):
    avg = []
    det = []
    with open(metric_dir + mdl_name + '.avg.metric.pkl', 'rb') as fd:
        avg = pickle.load(fd)
    with open(metric_dir + mdl_name + '.det.metric.pkl', 'rb') as fd:
        det = pickle.load(fd)
    return (avg, det)



mlp_1_32.compile(loss=build_masked_loss(keras.losses.categorical_crossentropy, MASK_VALUE), optimizer='adam')

# epochs : batch : est_time
# 100 : 10000 : 3h (6s/e)
# 100 : 1000  : 4h (8s/e)
# 100 : 100   : 13h(26s/e)
#originally tested with
#b:500
#e:400
b = 100
e = 10

mlp_1_32.fit(xtrn,ytrn, epochs=e, batch_size=b, callbacks=[mlp_1_32_metrics])

print(mlp_1_32_metrics.get_data())

with open('bacc_metric_simple.pkl', 'wb') as output:  # Overwrites any existing file.
    pickle.dump(mlp_1_32_metrics.get_data(), output, pickle.HIGHEST_PROTOCOL)
    
with open('bacc_metric_detailed.pkl', 'wb') as output:  # Overwrites any existing file.
    pickle.dump(mlp_1_32_metrics.get_detailed_data(), output, pickle.HIGHEST_PROTOCOL)




    
