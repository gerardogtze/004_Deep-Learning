import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


class Operation:
    def __init__(self, operation_type, bought_at, timestamp, n_shares,
                stop_loss, take_profit):
        self.operation_type = operation_type
        self.bought_at = bought_at
        self.timestamp = timestamp
        self.n_shares = n_shares
        self.sold_at = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit

def create_dataset(file):
    df = pd.read_csv(file)
    df[f'T_Minus_1'] = df['Close'].shift(1)
    df[f'T_Minus_2'] = df['Close'].shift(2)
    df[f'T_Minus_3'] = df['Close'].shift(3)
    df["T_Plus_5"] = df['Close'].shift(-5)
    df = df.dropna()  
    return df


def test_strategy(dataset, model_path):
    
    data = create_dataset(dataset)
    input_data = data[['Close', 'T_Minus_1', 'T_Minus_2', 'T_Minus_3']]
    model = load_model(model_path)
    predictions = model.predict(input_data)
    signals = (predictions > 0.54).astype(int)
    
    data["Position"] = signals
    
    cash = 1_000_000
    active_operations = []
    comision = 0.00125
    strategy_value = [1_000_000]
    
    for i, row in data.iterrows():
        #Cerrar las operaciones
        temp_operations = []
    
        for op in active_operations:
            
            #Cerrar las operaciones de perdida
            if op.stop_loss > row.Close:
                cash += row.Close * (1 - comision)
            
            #Cerrar las operaciones con ganancia
            elif op.take_profit < row.Close:
                cash += row.Close * (1 - comision)
            else:
                temp_operations.append(op)
        
        active_operations = temp_operations
        
        #Comprobar si tenemos suficiente dinero para comprar
        if cash > row.Close * (1 + comision):
            
            #Señal de compra
            if row.Position == 1:
            #if row.Close > row.Open:
                active_operations.append(Operation(operation_type ="long",
                                                bought_at = row.Close,
                                                timestamp = row.Date,
                                                n_shares= 1,
                                                stop_loss= row.Close * 0.95,
                                                take_profit= row.Close * 1.05)) #Guarda los datos en Operation
                cash -= row.Close * (1 + comision)
        
        #Calcular el valor de las posiciones abiertas
        total_value = len(active_operations) * row.Close
        strategy_value.append(cash + total_value)
    
    return strategy_value