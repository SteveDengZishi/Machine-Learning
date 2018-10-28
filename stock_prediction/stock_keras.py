#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:28:06 2018

@author: stevedeng
"""
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style 

df = pd.read_csv('JD.csv', parse_dates=True, index_col=1)
del df['Unnamed: 0']
print(df.head(10))
#Date is a datetime index
style.use('ggplot')
price_df = df.drop('Volume', axis=1)
price_df['100MA'] = price_df['Adj Close'].rolling(window=100).mean()
price_df['30MA'] = price_df['Adj Close'].rolling(window=30).mean()
ax1 = plt.subplot2grid((6,1),(0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(price_df.index, price_df['Adj Close'])
ax1.plot(price_df.index, price_df[['100MA','30MA']])
ax2.bar(df.index, df['Volume'])
plt.show()