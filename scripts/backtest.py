from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
import argparse
import tempfile
from functools import partial
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import training
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, Subset, DataLoader
import preprocess
import training_dataset
import rnn_models
from pathlib import Path
import time
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

import backtrader as bt


def simulate_trading(model, data, device, sequence_length):
    zeros_column = pd.Series([0]*len(data))
    class MyDataFeed(bt.feeds.GenericCSVData):

        lines = ('BID_PRICE_1', 'BID_SIZE_1', 'BID_PRICE_2', 'BID_SIZE_2', 'BID_PRICE_3', 'BID_SIZE_3',
             'ASK_PRICE_1', 'ASK_SIZE_1', 'ASK_PRICE_2', 'ASK_SIZE_2', 'ASK_PRICE_3', 'ASK_SIZE_3',
             'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond')

        params = (
            ('fromdate', None),      
            ('todate', None),        
            ('dtformat', '%Y-%m-%d'),     
            ('datetime', 0),         
            ('open', 4),                          
            ('high', 2),         
            ('low', 3),          
            ('close', 1),               
            ('volume', 5),        
            ('openinterest', 6),
            ('BID_PRICE_1', 7),
            ('BID_SIZE_1', 8),
            ('BID_PRICE_2', 9),
            ('BID_SIZE_2', 10),
            ('BID_PRICE_3', 11),
            ('BID_SIZE_3', 12),
            ('ASK_PRICE_1', 13),
            ('ASK_SIZE_1', 14),
            ('ASK_PRICE_2', 15),
            ('ASK_SIZE_2', 16),
            ('ASK_PRICE_3', 17),
            ('ASK_SIZE_3', 18), 
            ('year', 19),
            ('month', 20),
            ('day', 21),
            ('hour', 22),
            ('minute', 23),
            ('second', 24),
            ('microsecond', 25),

        )

    # Generic strategy - buy if prediction is higher than current price and sell if it is lower and we have a position
    class TestStrategy(bt.Strategy):

        def log(self, txt, dt=None):
            ''' Logging function fot this strategy'''
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

        def __init__(self):
            self.dataclose = self.datas[0].high
            self.inposition = False
            
            self.index = 0

        def next(self):
            startIdx = 2000
            if sequence_length > 500:
                startIdx = sequence_length * 4
            if self.index > 2000:
                bid_price_1 = [self.datas[0].__getattr__('BID_PRICE_1')[-i] for i in range(1, startIdx)]
                bid_size_1 = [self.datas[0].__getattr__('BID_SIZE_1')[-i] for i in range(1, startIdx)]
                bid_price_2 = [self.datas[0].__getattr__('BID_PRICE_2')[-i] for i in range(1, startIdx)]
                bid_size_2 = [self.datas[0].__getattr__('BID_SIZE_2')[-i] for i in range(1, startIdx)]
                bid_price_3 = [self.datas[0].__getattr__('BID_PRICE_3')[-i] for i in range(1, startIdx)]
                bid_size_3 = [self.datas[0].__getattr__('BID_SIZE_3')[-i] for i in range(1, startIdx)]
                
                ask_price_1 = [self.datas[0].__getattr__('ASK_PRICE_1')[-i] for i in range(1, startIdx)]
                ask_size_1 = [self.datas[0].__getattr__('ASK_SIZE_1')[-i] for i in range(1, startIdx)]
                ask_price_2 = [self.datas[0].__getattr__('ASK_PRICE_2')[-i] for i in range(1, startIdx)]
                ask_size_2 = [self.datas[0].__getattr__('ASK_SIZE_2')[-i] for i in range(1, startIdx)]
                ask_price_3 = [self.datas[0].__getattr__('ASK_PRICE_3')[-i] for i in range(1, startIdx)]
                ask_size_3 = [self.datas[0].__getattr__('ASK_SIZE_3')[-i] for i in range(1, startIdx)]

                year = [self.datas[0].__getattr__('year')[-i] for i in range(1, startIdx)]
                month = [self.datas[0].__getattr__('month')[-i] for i in range(1, startIdx)]
                day = [self.datas[0].__getattr__('day')[-i] for i in range(1, startIdx)]
                hour = [self.datas[0].__getattr__('hour')[-i] for i in range(1, startIdx)]
                minute = [self.datas[0].__getattr__('minute')[-i] for i in range(1, startIdx)]
                second = [self.datas[0].__getattr__('second')[-i] for i in range(1, startIdx)]
                microsecond = [self.datas[0].__getattr__('microsecond')[-i] for i in range(1, startIdx)]
                

                data = {
                    # 'COLLECTION_TIME': collection_time,
                    'BID_PRICE_1': bid_price_1,
                    'BID_SIZE_1': bid_size_1,
                    'BID_PRICE_2': bid_price_2,
                    'BID_SIZE_2': bid_size_2,
                    'BID_PRICE_3': bid_price_3,
                    'BID_SIZE_3': bid_size_3,
                    'ASK_PRICE_1': ask_price_1,
                    'ASK_SIZE_1': ask_size_1,
                    'ASK_PRICE_2': ask_price_2,
                    'ASK_SIZE_2': ask_size_2,
                    'ASK_PRICE_3': ask_price_3,
                    'ASK_SIZE_3': ask_size_3,
                    'year': year,
                    'month': month,
                    'day': day,
                    'hour': hour,
                    'minute': minute,
                    'second': second,
                    'microsecond': microsecond,
                }

                base = pd.DataFrame(data)

                base['COLLECTION_TIME'] = pd.to_datetime(base[['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']])


                base['COLLECTION_TIME'] = base['COLLECTION_TIME'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                base = base.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond'])

                base['MESSAGE_ID'] = zeros_column
                base['MESSAGE_TYPE'] = zeros_column
                base['SYMBOL'] = zeros_column

                base = preprocess.process_data(base)

                
                X = base[-(sequence_length):].drop('Time_Delta', axis=1).values
                

                input_tensor = torch.tensor(X, dtype=torch.float32)

                # Reshape the input tensor if necessary (e.g., add batch dimension)
                x = input_tensor.unsqueeze(0)

                # # Perform inference
                with torch.no_grad():
                    model.eval()
                    x = x.float()
                    x = x.to(device)
                    y_hat = model(x)
                    y_hat = F.softmax(y_hat, dim=1)
                    y_pred = torch.argmax(y_hat, dim=1)

                if y_pred[0] == 2 and self.inposition == False:
                    self.buy(size=100, exectype=bt.Order.Close)
                    self.inposition = True
                elif y_pred[0] == 1 and self.inposition == True:
                    self.sell(size=100, exectype=bt.Order.Close)
                    self.inposition = False


            self.index += 1


    
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    data = MyDataFeed(dataname='backtestview.csv')

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    cerebro.broker.setcash(100000.0)
        
    cerebro.run()

    end_portfolio_value = cerebro.broker.getvalue()

    return end_portfolio_value



def backtest_model(model, data_path, device, sequence_length):
    portfolio_value = 100000
    t1 = time.time()
    for file in data_path:
        temp = pd.read_csv(file)
        section_size = 200000
        num_sections = (len(temp) // section_size) + 1

        for i in range(num_sections):
            start_idx = i * section_size
            end_idx = min((i + 1) * section_size, len(temp))
            copy = temp.iloc[start_idx:end_idx]
        
            
            
            df = pd.read_csv(file)
            df = df.iloc[start_idx:end_idx]
            df2 = preprocess.cal_mid_price(df)

            collection = copy["COLLECTION_TIME"]

            copy = copy.loc[:, 'BID_PRICE_1':]


            num_rows = len(copy)
            start_date = pd.to_datetime('1700-01-01')
            copy['date'] = pd.date_range(start=start_date, periods=num_rows, freq='D').strftime('%Y-%m-%d')


            copy = copy[['date'] + [col for col in copy.columns if col != 'date']]
            
            copy.insert(1, 'MID_PRICE', df2.shift(1))

            zeros_column = pd.Series([0]*len(df))


            copy.insert(2, 'high', zeros_column)
            copy.insert(3, 'low', zeros_column)
            copy.insert(4, 'close', zeros_column)
            copy.insert(5, 'volume', zeros_column)
            copy.insert(6, 'openinterest', zeros_column)

            copy['COLLECTION_TIME'] = collection

            copy['COLLECTION_TIME'] = copy['COLLECTION_TIME'].replace('', np.nan)
            copy['COLLECTION_TIME'] = pd.to_datetime(copy['COLLECTION_TIME'])

            copy['year'] = copy['COLLECTION_TIME'].dt.year
            copy['month'] = copy['COLLECTION_TIME'].dt.month
            copy['day'] = copy['COLLECTION_TIME'].dt.day
            copy['hour'] = copy['COLLECTION_TIME'].dt.hour
            copy['minute'] = copy['COLLECTION_TIME'].dt.minute
            copy['second'] = copy['COLLECTION_TIME'].dt.second
            copy['microsecond'] = copy['COLLECTION_TIME'].dt.microsecond

            copy['COLLECTION_TIME'] = copy['COLLECTION_TIME'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


            copy = copy.drop(columns=['COLLECTION_TIME'])


            copy.to_csv('backtestview.csv', index=False)

            portfolio_value += (simulate_trading(model, copy, device, sequence_length) - 100000)
            print('Current Portfolio Value after ' + str(file) + ':  %.2f' % portfolio_value)
            print('=' * 89)
    
    t2 = time.time()
    
    print('Final Portfolio Value: %.2f' % portfolio_value)
    print('Total Time to Backtest:  ' + str(t2 - t1))
    print('=' * 89)
    print("----- End of backtesting -----")
    return portfolio_value
        

