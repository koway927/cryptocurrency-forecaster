import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def cal_mid_price(df):
    return (df['BID_PRICE_1'] + df['ASK_PRICE_1']) / 2

def cal_spread(df):
    return df['ASK_PRICE_1'] - df['BID_PRICE_1']

def cal_ask_price_mean(df):
    return df[['ASK_PRICE_1', 'ASK_PRICE_2', 'ASK_PRICE_3']].mean(axis=1)

def cal_bid_price_mean(df):
    return df[['BID_PRICE_1', 'BID_PRICE_2', 'BID_PRICE_3']].mean(axis=1)

def cal_ask_vol_mean(df):
    return df[['ASK_SIZE_1', 'ASK_SIZE_2', 'ASK_SIZE_3']].mean(axis=1)

def cal_bid_vol_mean(df):
    return df[['BID_SIZE_1', 'BID_SIZE_2', 'BID_SIZE_3']].mean(axis=1)

def cal_price_accumulated_difference(df):
    return df['ASK_PRICE_1'] - df['BID_PRICE_1'] + df['ASK_PRICE_2'] - df['BID_PRICE_2'] + df['ASK_PRICE_3'] - df['BID_PRICE_3']

def cal_vol_accumulated_difference(df):
    return df['ASK_SIZE_1'] - df['BID_SIZE_1'] + df['ASK_SIZE_2'] - df['BID_SIZE_2'] + df['ASK_SIZE_3'] - df['BID_SIZE_3']

def cal_vol_imbalance(df):
    return df['BID_SIZE_1'] - df['ASK_SIZE_1']

def cal_bid_ask_spread_ratio(df):
    return df['SPREAD'] / df['MID_PRICE']

def process_data(df):
    df['MID_PRICE'] = cal_mid_price(df)
    df['SPREAD'] = cal_spread(df)
    df['ASK_PRICE_MEAN'] = cal_ask_price_mean(df)
    df['BID_PRICE_MEAN'] = cal_bid_price_mean(df)
    df['ASK_VOL_MEAN'] = cal_ask_vol_mean(df)
    df['BID_VOL_MEAN'] = cal_bid_vol_mean(df)
    df['PRICE_ACCUMULATED_DIFFERENCE'] = cal_price_accumulated_difference(df)
    df['VOL_ACCUMULATED_DIFFERENCE'] = cal_vol_accumulated_difference(df)
    df["COLLECTION_TIME"] = pd.to_datetime(df["COLLECTION_TIME"]).values.astype(np.int64)
    df['VOLUME_IMBALANCE'] = cal_vol_imbalance(df)
    df['BID_ASK_SPREAD_RATIO'] = cal_bid_ask_spread_ratio(df)
    df['Time_Delta'] = df['COLLECTION_TIME'].diff()
    df.drop(["COLLECTION_TIME", "MESSAGE_ID", "MESSAGE_TYPE", "SYMBOL"], axis=1, inplace=True)
    # Remove outliers from each column
    columns_to_check_outliers = ['BID_PRICE_1', 'ASK_PRICE_1', 'BID_PRICE_2', 'ASK_PRICE_2', 'BID_PRICE_3', 'ASK_PRICE_3', 'MID_PRICE']
    Q1 = df[columns_to_check_outliers].quantile(0.25)
    Q3 = df[columns_to_check_outliers].quantile(0.75)
    IQR = Q3 - Q1
    for column in columns_to_check_outliers:
        median = df[column].median()
        df[column] = df[column].where(((df[column] >= (Q1[column] - 1.5 * IQR[column])) & (df[column] <= (Q3[column] + 1.5 * IQR[column]))), median)
    
    # Scale the selected columns
    columns_to_scale = [col for col in df.columns if col != 'Time_Delta']
    minmax_scaler = MinMaxScaler()
    df[columns_to_scale] = minmax_scaler.fit_transform(df[columns_to_scale])
    return df

