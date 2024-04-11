import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import pickle
import dnn
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense

def create_dataset(file):
    df = pd.read_csv(file)
    df[f'T_Minus_1'] = df['Close'].shift(1)
    df[f'T_Minus_2'] = df['Close'].shift(2)
    df[f'T_Minus_3'] = df['Close'].shift(3)
    df["T_Plus_5"] = df['Close'].shift(-5)
    df = df.dropna()  
    return df

def model_data(df, trade):
    x = df[['Close', 'T_Minus_1', 'T_Minus_2', 'T_Minus_3']]
    y = df['Close'] < df['T_Plus_5']
    
    if trade == "short":
        y = df['Close'] > df['T_Plus_5']
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1, shuffle=False)
    
    return X_train, X_test, y_train, y_test

def rnn_model(X_train, X_test, y_train, y_test, best_params):
    model = Sequential()
    model.add(LSTM(best_params["units_layer_1"], activation='relu', return_sequences=True, input_shape=(10, 1)))
    model.add(LSTM(best_params["units_layer_2"], activation='relu'))  
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy, model

def objective_rnn_wrapper(X_train, X_test, y_train, y_test):
    def objective_rnn(trial):
        model = Sequential()
        model.add(LSTM(trial.suggest_int('units_layer_1', 32, 256), activation='relu', return_sequences=True, input_shape=(10, 1)))
        model.add(LSTM(trial.suggest_int('units_layer_2', 32, 256), activation='relu'))  
        
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
        loss, accuracy = model.evaluate(X_test, y_test)
        return accuracy
    return objective_rnn