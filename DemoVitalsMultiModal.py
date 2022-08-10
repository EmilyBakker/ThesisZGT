
from asyncio import FastChildWatcher
from tabnanny import verbose
import pandas as pd
import numpy as np
from CustomDataGenerator import DemoVitalsDataGenerator
from CustomLosses import custom_binary_entropy_loss
from ModelBuilders import METRICS
from result_saver import save_cross_validation_results, save_learning_curve, save_multimodal_results, save_results
from roc_curve_generator import save_roc_curve

import tensorflow.python.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.python.keras.metrics import AUC, Precision, Recall
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras import layers
from tensorflow.keras.utils import plot_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import sys 
import os

K.clear_session()

#make file since there was an issue with writing a new file
def create_weights_file():
    """checks if the etag cache file exists. If the file does not exists, it is created"""
    if not os.path.isfile(f"/mnt/data/embakker/results/weights/dv_multi/weights_{experiment_name}.hdf5"):
        os.mknod(f"/mnt/data/embakker/results/weights/dv_multi/weights_{experiment_name}.hdf5")
def create_log_file():
    """checks if the etag cache file exists. If the file does not exists, it is created"""
    if not os.path.isfile(f"/mnt/data/embakker/results/logs/dv_multi/{experiment_name}.csv"):
        os.mknod(f"/mnt/data/embakker/results/logs/dv_multi/{experiment_name}.csv")

########################################################################################################
# Settings
########################################################################################################

model = 2

experiment_name = 'dv_model_freeze_ef_5'
experiment_notes = 'Freezing | early fusion | val_auc optimized | iter 5'

checkpoint_save_path = f'/mnt/data/embakker/results/weights/dv_multi/weights_{experiment_name}.hdf5'
weights_save_path = f'/mnt/data/embakker/results/weights/dv_multi/weights_{experiment_name}.hdf5'


# set 'early' or 'late'
fusion_strat = 'early'


learning_rate = 5e-3
comp_importance = 0.2 # Relative importance of mortality compared to complications, between 0 and 1
optimize_var = 'val_auc'
n_iter = 1
model = 5
iter = 5


evaluate_single_models = True
freeze_pretrained_models = True
validation_only = False
build_only = False

use_pretrained_weights= False
pretrained_model_path = '/mnt/data/embakker/results/model_weights/pretrained_multimodal_model.hdf5'

LABELDIR = '/mnt/data/embakker/data'
DATADIR = '/mnt/data/embakker/data'

fileout = f'/mnt/data/embakker/results/models/dv_multi/{experiment_name}.csv'
result_path = f'mnt/data/embakker/results/models/dv_multi/'
best_auc = 0
best_pr = 0

for iteration in range(n_iter):
    ########################################################################################################
    # Metrics
    ########################################################################################################

    ALLMETRICS = [
        AUC(name='auc'),
        Recall(name='recall'),
        Precision(name='precision'),
        AUC(name='pr', curve='PR')]


    ########################################################################################################
    # Pre-trained model paths
    ########################################################################################################

    WEIGHTSDIR = '/mnt/data/embakker/results/weights'
    path_vitals_model = f'{WEIGHTSDIR}/weights_Vitals_final_test.hdf5' 
    path_demo_model = f'{WEIGHTSDIR}/Demo/weights_Demo_1layers_2units.hdf5' 


    ########################################################################################################
    # Load pretrained models
    ########################################################################################################
    custom_objs = {'LeakyReLU': layers.LeakyReLU(),
        'inner_loss': custom_binary_entropy_loss(class_weights=None)} 

    vitals_model:   Model = load_model(path_vitals_model, custom_objects=custom_objs)
    demo_model:     Model = load_model(path_demo_model, custom_objects=custom_objs)

    ########################################################################################################
    # Build multimodal model
    ########################################################################################################

    print(f'iteration{iteration}_1')

    if fusion_strat == 'early':
        feature_layer_idx = -2
    elif fusion_strat == 'late':
        feature_layer_idx = -1
    else:
        raise NotImplementedError()

    

    # VITALS
    vitals_input = layers.Input(shape=(None, 3), name='vitals')
    vitals_inner = Model(inputs=vitals_model.input, outputs=vitals_model.layers[feature_layer_idx].output, name='vitals_model')
    vitals_out = vitals_inner(vitals_input)

    # DEMO
    demo_input = layers.Input(shape=(None,5), name='demo')
    demo_inner = Model(inputs=demo_model.input, outputs=demo_model.layers[feature_layer_idx].output, name='demo_model')
    demo_output = demo_inner(demo_input)

    # Combine models
    x = layers.concatenate([vitals_out, demo_output], name='concat')


    if fusion_strat == 'early':
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal', name='dense_0')(x)
        x = layers.LeakyReLU()(x)
        

    comp_pred = layers.Dense(1, activation='sigmoid', name='complication')(x)
    final_model = Model(inputs=[vitals_input, demo_input], outputs=[comp_pred])

    print(f'iteration{iteration}_2')
    # Optionally use pretrained weights
    if use_pretrained_weights:
        final_model.load_weights(pretrained_model_path, by_name=True)


    # Optinally freeze layers

    if freeze_pretrained_models:
        final_model.get_layer('demo_model').trainable = False
        final_model.get_layer('vitals_model').trainable = False

    else:
        final_model.get_layer('demo_model').trainable = True
        final_model.get_layer('vitals_model').trainable = True


    ########################################################################################################
    # Print and plot architectures
    ########################################################################################################
    print(f'iteration{iteration}_3')
    print("Demographics Model ",demo_model.summary())
    print("next")
    print("Vitals Model ", vitals_model.summary())
    print("next")
    print("DemoVitalsModal Model ", final_model.summary())

    print(f'iteration{iteration}_4')
    FIGOUT = f'/mnt/data/embakker/results/models/dv_multi/'
    
    plot_model(demo_model, to_file=f'{FIGOUT}architecture_demo.png')
    plot_model(vitals_model, to_file=f'{FIGOUT}architecture_vitals.png')
    plot_model(final_model, to_file=f'{FIGOUT}architecture_{experiment_name}.png')


    ########################################################################################################
    # Load data and create generators
    ########################################################################################################

    demo_train = pd.read_csv("/mnt/data/embakker/data/demo_train.csv", sep="|")
    demo_val = pd.read_csv("/mnt/data/embakker/data/demo_val.csv", sep="|")
    demo_test = pd.read_csv("/mnt/data/embakker/data/demo_test.csv", sep="|")

    vitals_train = pd.read_csv("/mnt/data/embakker/data/vitals_train.csv", sep="|")
    vitals_val = pd.read_csv("/mnt/data/embakker/data/vitals_val.csv", sep="|")
    vitals_test = pd.read_csv("/mnt/data/embakker/data/vitals_test.csv", sep="|")

    y_train = pd.read_csv("/mnt/data/embakker/data/label_train.csv", sep="|")
    y_val = pd.read_csv("/mnt/data/embakker/data/label_val.csv", sep="|")
    y_test = pd.read_csv("/mnt/data/embakker/data/label_test.csv", sep="|")

    
    train_generator = DemoVitalsDataGenerator(label_data = y_train,
                                            demo_data = demo_train,
                                            vitals_data = vitals_train,
                                            shuffle=True)

    val_generator = DemoVitalsDataGenerator(label_data = y_val,
                                            demo_data = demo_val,
                                            vitals_data = vitals_val,
                                            shuffle=True)

    test_generator = DemoVitalsDataGenerator(label_data = y_test,
                                            demo_data = demo_test,
                                            vitals_data = vitals_test,
                                            shuffle=True)


    ########################################################################################################
    # Callbacks
    ########################################################################################################

    create_log_file()

    early_stop = EarlyStopping(monitor=optimize_var, mode = 'max', patience=12, verbose=1, min_delta=1e-4, restore_best_weights = True)
    reduce_lr = ReduceLROnPlateau(monitor=optimize_var, mode = 'max', factor=0.5, patience=5, verbose=1, min_delta=1e-4)
    csv_logger = CSVLogger(f'/mnt/data/embakker/results/logs/dv_multi/log_{experiment_name}.csv', append=True, separator=';')
    checkpointer = ModelCheckpoint(filepath=checkpoint_save_path, verbose=1, save_best_only=True, monitor=optimize_var, mode='max')


    ########################################################################################################
    # Configure loss function
    ########################################################################################################

    custom_loss = custom_binary_entropy_loss()

    
    ########################################################################################################
    # Evaluate single models
    ########################################################################################################


    optimizer = Adam(learning_rate=learning_rate)
    if evaluate_single_models:
        # Set generators to single outputs
        #val_generator.multilabel = False
        #test_generator.multilabel = False

        # VITALS
        vitals_eval = Model(inputs=vitals_input, outputs=vitals_model(vitals_input))
        vitals_eval.compile(optimizer=optimizer, metrics=ALLMETRICS, loss='binary_crossentropy')
        vitals_val_results = vitals_eval.evaluate(val_generator)
        vitals_test_results = vitals_eval.evaluate(test_generator)

        # DEMO
        demo_eval = Model(inputs=demo_input, outputs=demo_model(demo_input))
        demo_eval.compile(optimizer=optimizer, metrics=ALLMETRICS, loss='binary_crossentropy')
        demo_val_results = demo_eval.evaluate(val_generator)
        demo_test_results = demo_eval.evaluate(test_generator)

        print('### STANDALONE RESULTS ###')
        for i, name in enumerate(demo_eval.metrics_names):
            if name == 'loss': continue
            print('#'*40)
            print('{:10s} {:>10s} {:>10s}'.format(name, 'val', 'test'))
            print('{:10s} {:10.3f} {:10.3f}'.format('vitals', vitals_val_results[i], vitals_test_results[i]))
            print('{:10s} {:10.3f} {:10.3f}'.format('demo', demo_val_results[i], demo_test_results[i]))
        print('#'*40)

        

    ########################################################################################################
    # Optionally only build the model
    ########################################################################################################
    create_weights_file()
    if build_only:

        final_model.save(weights_save_path)
        print('Saved model, exiting...')
        sys.exit()


    ########################################################################################################
    # Fit multimodal model
    ########################################################################################################
    print(f'iteration{iteration}_5')
    optimizer = Adam(learning_rate=learning_rate)
    final_model.compile(loss='binary_crossentropy', 
                        optimizer=optimizer, 
                        metrics= ALLMETRICS,#, 'complication': COMP_METRICS
                        #loss_weights={'mortality': 1.} #, 'complication': comp_importance
                    )

    history = final_model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=100,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[early_stop, reduce_lr, csv_logger, checkpointer])


    train_results = final_model.evaluate(train_generator, verbose=0)
    val_results = final_model.evaluate(val_generator, verbose=0)
    test_results = final_model.evaluate(test_generator, verbose=0)

    temp_auc = val_results[final_model.metrics_names.index('auc')]
    temp_pr = val_results[final_model.metrics_names.index('pr')]

    # Save model weights if best at the moment
    if temp_auc > best_auc:
        best_auc = temp_auc
    #    final_model.save(weights_save_path)

    #Save model weights if best at the moment
    if temp_pr > best_pr:
        best_pr = temp_pr
    #    final_model.save(weights_save_path)

        
    results_df = pd.DataFrame(columns=['split'] + [name for name in final_model.metrics_names])

    results_df.loc[len(results_df.index)] = ['train'] + train_results
    results_df.loc[len(results_df.index)] = ['val'] + val_results
    results_df.loc[len(results_df.index)] = ['test'] + test_results

    print(f'iteration{iteration}_6')




print('DONE')
print(f'FINAL BEST:   AUC={best_auc:.3f}')
print(f'FINAL BEST:   PR={best_pr:.3f}')
print('#'*40)

# ########################################################################################################
# # Save results
# ########################################################################################################

results_df.to_csv(fileout, sep=';', index=False)

