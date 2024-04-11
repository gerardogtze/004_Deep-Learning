import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import optuna
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam



def objective_dnn_wrapper(X_train, X_test, y_train, y_test):
    
    def objective_dnn(trial):        
        n_layers = trial.suggest_int('n_layers', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        
        model = Sequential()
        model.add(layers.Dense(trial.suggest_int('units_layer_0', 32, 256), activation='relu', input_shape=(X_train.shape[1],)))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))

        for i in range(1, n_layers):
            model.add(layers.Dense(trial.suggest_int(f'units_layer_{i}', 32, 256), activation='relu'))
            model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        
        model.add(layers.Dense(1, activation='sigmoid'))  # Assuming binary classification
        
        optimizer = Adam()
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        
        predictions = model.predict(X_test)
        pred_labels = np.where(predictions > 0.5, 1, 0)
        loss, accuracy = model.evaluate(X_test, y_test)
        
        return accuracy
    return objective_dnn

def deep_neural_network(X_train, X_test, y_train, y_test, best_params):
            
    model = Sequential()
    model.add(layers.Dense(best_params["units_layer_0"], activation='relu', input_shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dropout(rate=best_params["dropout_rate"]))
    
    for i in range(1, best_params["n_layers"]):
        model.add(layers.Dense(best_params[f'units_layer_{i}'], activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=best_params["dropout_rate"]))
        
    model.add(layers.Dense(1, activation='sigmoid'))  # Assuming binary classification
        
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        
    predictions = model.predict(X_test)
    pred_labels = np.where(predictions > 0.5, 1, 0)
    loss, accuracy = model.evaluate(X_test, y_test)
        
    
    return accuracy, model
