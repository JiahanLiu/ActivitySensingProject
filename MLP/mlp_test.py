import numpy as np
import pickle
from project_share import *
import tensorflow as tf
from tensorflow import keras

mlp_input_dim = 175
model_dir = "./models/"
metric_dir = "./metrics/"
#create models
linear = keras.Sequential()
mlp_1_32 = keras.Sequential()
mlp_1_256 = keras.Sequential()
mlp_2_16_16 = keras.Sequential()
mlp_2_8_16 = keras.Sequential()
mlp_2_16_8 = keras.Sequential()
mlp_2_16_32 = keras.Sequential()
mlp_2_32_16 = keras.Sequential()
mlp_3_8_8_8 = keras.Sequential()
mlp_3_16_16_16 = keras.Sequential()
mlp_3_32_32_32 = keras.Sequential()
mlp_3_8_16_8 = keras.Sequential()
mlp_3_32_16_32 = keras.Sequential()
mlp_3_16_32_16 = keras.Sequential()
mlp_3_32_64_32 = keras.Sequential()
mlp_5_16_16_16_16_16 = keras.Sequential()
mlp_5_16_32_64_32_16 = keras.Sequential()
mlp_5_64_32_16_32_64 = keras.Sequential()

linear.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_1_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_1_32.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_1_256.add(keras.layers.Dense(256, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_1_256.add(keras.layers.Dense(51, input_dim=mlp_input_dim, kernel_initializer='normal', activation='sigmoid'))

mlp_2_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_16_16.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_2_16_16.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_2_8_16.add(keras.layers.Dense(8, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_8_16.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_2_8_16.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_2_16_8.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_16_8.add(keras.layers.Dense(8, kernel_initializer='normal', activation='relu'))
mlp_2_16_8.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_2_16_32.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_16_32.add(keras.layers.Dense(32, kernel_initializer='normal', activation='relu'))
mlp_2_16_32.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_2_32_16.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_2_32_16.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_2_32_16.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_3_8_8_8.add(keras.layers.Dense(8, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_8_8_8.add(keras.layers.Dense(8, kernel_initializer='normal', activation='relu'))
mlp_3_8_8_8.add(keras.layers.Dense(8, kernel_initializer='normal', activation='relu'))
mlp_3_8_8_8.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_3_16_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_16_16_16.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_3_16_16_16.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_3_16_16_16.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_3_32_32_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_32_32_32.add(keras.layers.Dense(32, kernel_initializer='normal', activation='relu'))
mlp_3_32_32_32.add(keras.layers.Dense(32, kernel_initializer='normal', activation='relu'))
mlp_3_32_32_32.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_3_8_16_8.add(keras.layers.Dense(8, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_8_16_8.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_3_8_16_8.add(keras.layers.Dense(8, kernel_initializer='normal', activation='relu'))
mlp_3_8_16_8.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_3_32_16_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_32_16_32.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_3_32_16_32.add(keras.layers.Dense(32, kernel_initializer='normal', activation='relu'))
mlp_3_32_16_32.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_3_16_32_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_16_32_16.add(keras.layers.Dense(32, kernel_initializer='normal', activation='relu'))
mlp_3_16_32_16.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_3_16_32_16.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_3_32_64_32.add(keras.layers.Dense(32, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_3_32_64_32.add(keras.layers.Dense(64, kernel_initializer='normal', activation='relu'))
mlp_3_32_64_32.add(keras.layers.Dense(32, kernel_initializer='normal', activation='relu'))
mlp_3_32_64_32.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_5_16_16_16_16_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_16_16_16_16_16.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_5_16_16_16_16_16.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_5_16_16_16_16_16.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_5_16_16_16_16_16.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_5_16_16_16_16_16.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_5_16_32_64_32_16.add(keras.layers.Dense(16, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_16_32_64_32_16.add(keras.layers.Dense(32, kernel_initializer='normal', activation='relu'))
mlp_5_16_32_64_32_16.add(keras.layers.Dense(64, kernel_initializer='normal', activation='relu'))
mlp_5_16_32_64_32_16.add(keras.layers.Dense(32, kernel_initializer='normal', activation='relu'))
mlp_5_16_32_64_32_16.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_5_16_32_64_32_16.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_5_64_32_16_32_64.add(keras.layers.Dense(64, input_dim=mlp_input_dim, kernel_initializer='normal', activation='relu'))
mlp_5_64_32_16_32_64.add(keras.layers.Dense(32, kernel_initializer='normal', activation='relu'))
mlp_5_64_32_16_32_64.add(keras.layers.Dense(16, kernel_initializer='normal', activation='relu'))
mlp_5_64_32_16_32_64.add(keras.layers.Dense(32, kernel_initializer='normal', activation='relu'))
mlp_5_64_32_16_32_64.add(keras.layers.Dense(64, kernel_initializer='normal', activation='relu'))
mlp_5_64_32_16_32_64.add(keras.layers.Dense(51, kernel_initializer='normal', activation='sigmoid'))

mlp_models = {
"linear"                : linear,
"mlp_1_32"              : mlp_1_32,
"mlp_1_256"             : mlp_1_256,
"mlp_2_16_16"           : mlp_2_16_16,
"mlp_2_8_16"            : mlp_2_8_16,
"mlp_2_16_8"            : mlp_2_16_8,
"mlp_2_16_32"           : mlp_2_16_32,
"mlp_2_32_16"           : mlp_2_32_16,
"mlp_3_8_8_8"           : mlp_3_8_8_8,
"mlp_3_16_16_16"        : mlp_3_16_16_16,
"mlp_3_32_32_32"        : mlp_3_32_32_32,
"mlp_3_8_16_8"          : mlp_3_8_16_8,
"mlp_3_32_16_32"        : mlp_3_32_16_32,
"mlp_3_16_32_16"        : mlp_3_16_32_16,
"mlp_3_32_64_32"        : mlp_3_32_64_32,
"mlp_5_16_16_16_16_16"  : mlp_5_16_16_16_16_16,
"mlp_5_16_32_64_32_16"  : mlp_5_16_32_64_32_16,
"mlp_5_64_32_16_32_64"  : mlp_5_64_32_16_32_64
}

linear_metrics              = Project_Metrics(mtst)
mlp_1_32_metrics            = Project_Metrics(mtst)
mlp_1_256_metrics           = Project_Metrics(mtst)
mlp_2_16_16_metrics         = Project_Metrics(mtst)
mlp_2_8_16_metrics          = Project_Metrics(mtst)
mlp_2_16_8_metrics          = Project_Metrics(mtst)
mlp_2_16_32_metrics         = Project_Metrics(mtst)
mlp_2_32_16_metrics         = Project_Metrics(mtst)
mlp_3_8_8_8_metrics         = Project_Metrics(mtst)
mlp_3_16_16_16_metrics      = Project_Metrics(mtst)
mlp_3_32_32_32_metrics      = Project_Metrics(mtst)
mlp_3_8_16_8_metrics        = Project_Metrics(mtst)
mlp_3_32_16_32_metrics      = Project_Metrics(mtst)
mlp_3_16_32_16_metrics      = Project_Metrics(mtst)
mlp_3_32_64_32_metrics      = Project_Metrics(mtst)
mlp_5_16_16_16_16_16_metrics= Project_Metrics(mtst)
mlp_5_16_32_64_32_16_metrics= Project_Metrics(mtst)
mlp_5_64_32_16_32_64_metrics= Project_Metrics(mtst)

mlp_metrics = {
"linear"                : linear_metrics,
"mlp_1_32"              : mlp_1_32_metrics,
"mlp_1_256"             : mlp_1_256_metrics,
"mlp_2_16_16"           : mlp_2_16_16_metrics,
"mlp_2_8_16"            : mlp_2_8_16_metrics,
"mlp_2_16_8"            : mlp_2_16_8_metrics,
"mlp_2_16_32"           : mlp_2_16_32_metrics,
"mlp_2_32_16"           : mlp_2_32_16_metrics,
"mlp_3_8_8_8"           : mlp_3_8_8_8_metrics,
"mlp_3_16_16_16"        : mlp_3_16_16_16_metrics,
"mlp_3_32_32_32"        : mlp_3_32_32_32_metrics,
"mlp_3_8_16_8"          : mlp_3_8_16_8_metrics,
"mlp_3_32_16_32"        : mlp_3_32_16_32_metrics,
"mlp_3_16_32_16"        : mlp_3_16_32_16_metrics,
"mlp_3_32_64_32"        : mlp_3_32_64_32_metrics,
"mlp_5_16_16_16_16_16"  : mlp_5_16_16_16_16_16_metrics,
"mlp_5_16_32_64_32_16"  : mlp_5_16_32_64_32_16_metrics,
"mlp_5_64_32_16_32_64"  : mlp_5_64_32_16_32_64_metrics
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


run = True
print("am I running?")
print(run)
if run:
    #compile models
    for key in mlp_models:
        model = mlp_models[key]
        model.compile(loss=build_masked_loss(keras.losses.categorical_crossentropy, MASK_VALUE), optimizer='adam')

    # epochs : batch : est_time
    # 100 : 10000 : 3h (6s/e)
    # 100 : 1000  : 4h (8s/e)
    # 100 : 100   : 13h(26s/e)
    #originally tested with
    #b:500
    #e:400
    b = 500
    e = 400

    for key in mlp_models:
        model = mlp_models[key]
        metric = mlp_metrics[key]
        
        print("training " + key)
        
        model.fit(xtrn,ytrn, epochs=e, batch_size=b, callbacks=[metric])
        mlp_metrics[key] = metric
        
        with open(metric_dir + key + '.avg.metric.pkl', 'wb') as output:
            pickle.dump(metric.get_data(), output, pickle.HIGHEST_PROTOCOL)
            
        with open(metric_dir + key + '.det.metric.pkl', 'wb') as output:
            pickle.dump(metric.get_detailed_data(), output, pickle.HIGHEST_PROTOCOL)
        
        model.save(model_dir + key + '.mdl')