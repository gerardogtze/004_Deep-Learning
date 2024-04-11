import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import pickle
import dnn
import rnn
import os


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

model_results = []
folder_path = '../data'

results= {}

for filename in os.listdir(folder_path):
    
    file_path = os.path.join(folder_path, filename)
    positions = ["long","short"]
    
    for position in positions:
        dataset = create_dataset(file_path)
        X_train, X_test, y_train, y_test = model_data(dataset, position)
        study = optuna.create_study(direction='maximize')
        objective = rnn.objective_rnn_wrapper(X_train, X_test, y_train, y_test)
        study.optimize(objective, n_trials=1)
        best_score = study.best_value
        accuracy, model = rnn.rnn_model(X_train, X_test, y_train, y_test, study.best_params)
        new_name = filename.replace(".csv","_" + position)
        results[new_name] = accuracy
        model.save("machine_learning/models/" + "rnn_" + new_name + ".keras")
       

for key, value in results.items():
    print(f"Dataset: {key}, Accuracy: {value}")     
