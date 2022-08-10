#################
#LIBRARIES
#################

from statistics import mode
from CustomDataGenerator import DemoDataGenerator
from CustomLosses import custom_binary_entropy_loss
from LastDataCleaning import Y_val

import numpy as np
from ModelBuilders import build_dense_model, build_lstm
from result_saver import save_learning_curve, save_results
from roc_curve_generator import save_roc_curve

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import tensorflow.python.keras.backend as K

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score


import sys

K.clear_session()

#################
# SETUP
#################



experiment_name = 'Demo_dense_experiment'
experiment_notes = 'Demo | pr optimized | 1 layers  | Dense output 2'

validation_only = False
document_results = True



y_train = pd.read_csv("/mnt/data/embakker/data/label_train.csv", sep="|")
y_val = pd.read_csv("/mnt/data/embakker/data/label_val.csv", sep="|")
y_test = pd.read_csv("/mnt/data/embakker/data/label_test.csv", sep="|")


x_train = pd.read_csv("/mnt/data/embakker/data/demo_train.csv", sep="|")
x_val = pd.read_csv("/mnt/data/embakker/data/demo_val.csv", sep="|")
x_test = pd.read_csv("/mnt/data/embakker/data/demo_test.csv", sep="|")




optimization_var = 'val_pr'
num_features = 5
batch_size = 16
shuffle = True
bidirectional=False
lstm_units = 10
lstm_layers = 1
dense_units = [2, 2]
unit_dropout = 0.5
dense_dropout = 0.5
balance_classes = False
label_smoothing = 0
cweight_dict = None

#################
# CREATE DATA GENERATORS
#################

train_generator = DemoDataGenerator(label_data=y_train,demo_data= x_train, batch_size=batch_size, shuffle=shuffle)
val_generator = DemoDataGenerator(y_val, x_val, batch_size=batch_size, shuffle=shuffle)
test_generator = DemoDataGenerator(y_test, x_test, batch_size=batch_size, shuffle=shuffle)


# Custom loss function
custom_loss = custom_binary_entropy_loss(class_weights=cweight_dict, label_smoothing=label_smoothing)
    #custom_loss = pr_loss(class_weights=cweight_dict)

#################
# BUILD MODEL
#################


model = build_lstm(lstm_units=lstm_units, lstm_layers=lstm_layers, dense_units=dense_units, bidirectional=bidirectional, num_features=num_features, 
                   u_dropout=unit_dropout, dense_dropout=dense_dropout, target_replication=use_tr, loss=custom_loss,
                   learning_rate=5e-4)




print(experiment_name)
print(experiment_notes)

###################################################################
# Callback functions

early_stop = EarlyStopping(monitor=optimization_var, mode = 'max', patience=10, verbose=1, min_delta=1e-4, restore_best_weights = True)
reduce_lr = ReduceLROnPlateau(monitor=optimization_var, mode = 'max', factor=0.5, patience=5, verbose=1, min_delta=1e-4)
csv_logger = CSVLogger(f'/mnt/data/embakker/results/logs/log_{experiment_name}.csv', append=True, separator=';')
weights_save_path = (f'/mnt/data/embakker/results/weights/weights_{experiment_name}.hdf5')
checkpointer = ModelCheckpoint(filepath=weights_save_path, verbose=1, save_best_only=True, monitor=optimization_var, mode='max')


###################################################################

print(f'FINISHED SETUP of DEMO model')



#################
# FIT MODEL
#################


history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stop, reduce_lr, csv_logger, checkpointer])


###################################################################

print('FINISHED TRAINING; saving results...')

#################
# SAVE RESULTS
#################

if not document_results: sys.exit()

file_name_to_record_results = f'/mnt/data/embakker/results/models/demo/{experiment_name}.csv'

# saving validation results
threshold = 0.05


y_true = val_generator.get_true_labels()
y_predict_scores = model.predict(val_generator,verbose = 1)
y_predict = y_predict_scores >= threshold


save_learning_curve(experiment_name, history)
#save_roc_curve(experiment_name=f'{experiment_name}_val', y_true = y_true, y_predict = y_predict_scores )
save_results(file_name_to_record_results, y_true = y_true, y_predict = y_predict, 
            y_predict_scores = y_predict_scores,
            experiment_name = experiment_name,
            experiment_notes = f'{experiment_notes} | Validation',
            classifier = 'LSTM')



# saving test results

if not validation_only:
    y_true = test_generator.get_true_labels()
    y_predict_scores = model.predict(test_generator,verbose = 1)
    y_predict = y_predict_scores >= threshold



    save_roc_curve(experiment_name=f'{experiment_name}_test', y_true = y_true, y_predict = y_predict_scores )

    save_results(file_name_to_record_results, y_true = y_true, y_predict = y_predict, 
                y_predict_scores = y_predict_scores,
                experiment_name = experiment_name,
                experiment_notes = f'{experiment_notes} | Test',
                classifier = 'LSTM')