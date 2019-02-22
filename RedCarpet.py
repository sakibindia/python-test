#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:44:10 2019

@author: Admin
"""

from datetime import date
from nsepy import get_history
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import holidays
from business_calendar import Calendar,MO,TU,WE,TH,FR


#function for moving average 
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')

def rolling_windows(dataset,rollingsize):
    window=np.repeat(1,int(rollingsize))
    return np.convolve(dataset,window,'same')
    



symbols=["INFY","TCS","NIFTY"]
# Stock options (Similarly for index options, set index = True) 
INFY= get_history(symbol="INFY",
                  start=date(2015,1,1),
                  end=date(2015,12,31)
                  )

TCS=get_history(symbol="TCS",
                start=date(2015,1,1),
                end=date(2015,12,31)
        )

NIFTY=get_history(symbol="NIFTY",
                  start=date(2015,1,1),
                  end=date(2015,12,31),
        )  


#TCS.to_csv('tcs_stock.csv', encoding='utf-8', index=False)
INFY.insert(0, 'Date',  pd.to_datetime(INFY.index,format='%Y-%m-%d') )
TCS.insert(0, 'Date',  pd.to_datetime(TCS.index,format='%Y-%m-%d') )
NIFTY.insert(0, 'Date',  pd.to_datetime(NIFTY.index,format='%Y-%m-%d') )
#convert all data into dataframe for furtherprcessing

INFY = pd.DataFrame(INFY)
TCS = pd.DataFrame(TCS)
NIFTY = pd.DataFrame(NIFTY)

print()


TCS["Date"] = pd.to_datetime(TCS["Date"],errors='ignore')
INFY["Date"] = pd.to_datetime(INFY["Date"],errors='ignore')
NIFTY["Date"] = pd.to_datetime(NIFTY["Date"],errors='ignore')


print(INFY.columns)
print(TCS.shape)
print(INFY.shape)
print(NIFTY.shape)



stocks = [TCS, INFY, NIFTY]


def make_features(dataset):
    dataset['Date'] = pd.to_datetime(dataset['Date'])   
    dataset['Year'] = dataset['Date'].dt.year
    dataset['Month'] = dataset.Date.dt.month
    dataset['Day'] = dataset.Date.dt.day
    dataset['WeekOfYear'] = dataset.Date.dt.weekofyear
    
    
#for i in range(len(stocks)):
 #   make_features(i)

weeks = [4, 16, 28, 40, 52]


def indexing(stock):
    stock.index = stock['Date']
    return stock

indexing(TCS)
indexing(INFY)
indexing(NIFTY)

        
    
TCS.name = 'TCS'
INFY.name = 'INFY'
NIFTY.name = 'NIFTY_IT'
#Create rolling window of size 10 on each stock/index
def plt_series(stock, weeks = [4, 16, 28, 40, 52]):
    
    dummy = pd.DataFrame()
    # First Resampling into Weeks format to calculate for weeks
    dummy['Close'] = stock['Close'].resample('W').mean() 
     
    for i in range(len(weeks)):
        moving_avg = dummy['Close'].rolling(weeks[i]).mean() # using inbuilt function
        dummy[" Mov.AVG for " + str(weeks[i])+ " Weeks"] = moving_avg
        print('Calculated Moving Averages: for {0} weeks: \n\n {1}' .format(weeks[i], dummy['Close']))
        dummy.plot(title="Moving Averages for {} \n\n" .format(stock.name))

plt_series(TCS)
plt_series(INFY)
plt_series(NIFTY)



def rolling_windows(dataset, win = [10, 75]):
    
    dummy = pd.DataFrame()
    
    dummy['Close'] = dataset['Close']
     
    for i in range(len(win)):
        moving_avg= dummy['Close'].rolling(win[i]).mean() # M.A using predefined function
        dummy[" Mov.AVG for " + str(win[i])+ " Roll Window"] = moving_avg
        print('Calculated Moving Averages: for {0} weeks: \n\n {1}' .format(win[i], dummy['Close']))
        dummy.plot(title="Moving Averages for {} \n\n" .format(dataset.name))


rolling_windows(TCS)
rolling_windows(INFY)
rolling_windows(NIFTY)


def volume_shocks(stock):
    stock["vol_t+1"] = stock.Volume.shift(1)  #next rows value
    
    stock["volume_shock"] = ((abs(stock["vol_t+1"] - stock["Volume"])/stock["Volume"]*100)  > 10).astype(int)
    
    return stock





volume_shocks(TCS)
volume_shocks(INFY)
volume_shocks(NIFTY)




#price Shock 

def price_shocks(stock):
    """
    'ClosePrice' - Close_t
    'Close Price next day - vol_t+1
    
    """
    stock["price_t+1"] = stock.Close.shift(1)  #next rows value
    
    stock["price_shock"] = (abs((stock["price_t+1"] - stock["Close"])/stock["Close"]*100)  > 2).astype(int)
    
    stock["price_black_swan"] = stock['price_shock'] # Since both had same data anad info/
    
    return stock



price_shocks(TCS)
price_shocks(INFY)
price_shocks(NIFTY)




