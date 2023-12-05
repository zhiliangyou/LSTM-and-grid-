# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 05:46:37 2023

@author: 35003
"""

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
para
'''

window = 20
grid_mul = 3
grid_num = 5
slippage = 0.0
brokerage = 0.0005 / 2
trans_cost = slippage + brokerage

shorter = 'CAD_1min.xlsx'
longer = 'CAD_15min.xlsx'

################################################################
'''
strat
'''
#grid
def consolidate_df(shorter, longer, window, grid_mul, grid_num):
    
    shorter_df = pd.read_excel(shorter)
    longer_df = pd.read_excel(longer)

    #only keep close
    shorter_df = shorter_df[['Dates', 'Close']]
    longer_df = longer_df[['Dates', 'Close']]

    # shift so we cant look into the future
    longer_df['Close'] = longer_df['Close'].shift(1)

    # rolling mean standard deviation
    ############# mean here should be replaced by the predicted series
    longer_df['roll_mean'] = longer_df['Close'].rolling(window).mean()
    longer_df['roll_std'] = longer_df['Close'].rolling(window).std()

    # Grid
    longer_df[f'grid({grid_num})'] = longer_df['roll_mean'] +  grid_mul * longer_df['roll_std']
    longer_df[f'grid({-grid_num})'] = longer_df['roll_mean'] -  grid_mul * longer_df['roll_std']
    for i in range(1, grid_num):
        longer_df[f'grid({i})'] = longer_df['roll_mean'] + i/grid_num * (longer_df[f'grid({grid_num})'] - longer_df['roll_mean'])
        longer_df[f'grid({-i})'] = longer_df['roll_mean'] - i/grid_num * (longer_df['roll_mean'] - longer_df[f'grid({-grid_num})'])

    # merge and return
    shorter_df.rename(columns={'Close':'1min'}, inplace=True)
    longer_df.rename(columns={'Close':'15min'}, inplace=True)

    return pd.merge_asof(shorter_df, longer_df.sort_values('Dates'), on='Dates')



#signal
def apply_strat(df, grid_num, stop_period = 20):
  
    df['signal'] = 1
    stop_trigger = 0

    for i in range(len(df)):
        # if price breaks out of range, we need to stop til stop_period
        if stop_trigger:
            df.loc[i, 'signal'] = 0
            stop_trigger = (stop_trigger + 1) % stop_period
        else:
            # price is ard our mid price
            if df.loc[i, 'grid(-1)'] < df.loc[i, '1min'] <= df.loc[i, 'grid(1)']:
                df.loc[i, 'signal'] = 0
            #price is out of our upper and lower limit
            elif df.loc[i, f'grid({grid_num})'] < df.loc[i, '1min'] or df.loc[i, '1min'] <= df.loc[i, f'grid({-grid_num})']:
                df.loc[i, 'signal'] = 0
                stop_trigger = (stop_trigger + 1) % stop_period
            else:
                for j in range(1, grid_num):
                    # price falls in positive grid j
                    if df.loc[i, f'grid({j})'] < df.loc[i, '1min'] <= df.loc[i, f'grid({j + 1})']:
                        df.loc[i, 'signal'] = -j
                    # negative grid
                    elif (df.loc[i, f'grid({-j - 1})'] < df.loc[i, '1min'] <= df.loc[i, f'grid({-j})']):
                        df.loc[i, 'signal'] = j

    
    df['B/S'] = df['signal'].diff().fillna(df['signal'])
    return df

#back_test
def backtest_result(df, grid_num = 5 , cost = 0.0, amount = 100000):
    
    df['quantity'] = round(amount * df['B/S'] / df['1min'] / (grid_num - 1), 0)

    df['position'] = 0
    df.loc[0,'position'] = df.loc[0,'quantity']
    for i in range(1, len(df)):
        df.loc[i,'position'] = df.loc[i - 1,'position'] + df.loc[i,'quantity']

    df['notional'] = -df['quantity'] * df['1min']

    df['cum_notional'] = 0
    df.loc[0,'cum_notional'] = df.loc[0,'notional']
    for i in range(1, len(df)):
        df.loc[i,'cum_notional'] = df.loc[i - 1,'cum_notional'] + df.loc[i,'notional']

    df['PnL'] = df['cum_notional'] + df['position'] * df['1min']
    df['1minPnL'] = df['PnL'].diff().fillna(0)
    df['trans_cost'] = abs(df['notional']) * cost
    df['net_PnL'] = df['PnL'] - df['trans_cost'].cumsum()
    df['net_1minPnL'] = df['net_PnL'].diff().fillna(0)
    return df



#stat
def key_stat(df):
    res = {}
    res['gross_return'] = df.loc[len(df) - 1,'PnL']
    res['num_of_trade'] = len(df[df['B/S'] != 0])
    res['trans_cost'] = df['trans_cost'].sum()
    res['net_return'] = df.loc[len(df) - 1,'net_PnL']
    res['Sharpe'] = df['1minPnL'].mean() / df['1minPnL'].std()
    return res




#############################################################################
'''
back test
'''
if __name__ == '__main__':
    df = consolidate_df(shorter, longer, window, grid_mul, grid_num)
    pd.set_option('display.precision',  4)
    #df.head(20)
    
    df = apply_strat(df, grid_num, 30)
    df = backtest_result(df, grid_num, trans_cost, 100000)
    pd.set_option('display.precision',  4)
    #df.head(20)
    
    res = key_stat(df)
    res
    
    #plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('1-min Price', fontsize=14)
    ax1.plot(df['Dates'], df['1min'], color = 'blue', label = '1min Close')
    ax1.legend(loc='best')
    
    ax2.set_ylabel('PnL', fontsize=14)
    ax2.plot(df['Dates'], df['PnL'], color = 'orange', label = 'PnL')
    ax2.legend(loc='best')
    
    fig.suptitle("Backtesting result of AAPL reference price = 20-SMA", fontsize=18)
    
    
