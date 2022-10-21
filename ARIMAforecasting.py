# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 10:51:02 2022
@author: Jerry Lu
@Email: jerry.lu@wustl.edu 
"""

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing as mp
import warnings 
import time
warnings.filterwarnings("ignore")


class RCforecast(object):
  def __init__(self, data, window, fc_period=1, gridSearch=10, frequency='W-MON'):
    self.data = data # 100 rows time-series data
    self.window = window # 50 rows 
    self.stp = fc_period # 2 rows
    self.gridSearch = gridSearch # 10 x 10
    self.freq = frequency # time-series data frequency

    self._col = data.columns[0]
    
  def eval_arima(self, data, pdq):
    """
    Apply ARIMA to forecast based on provided data and (p,d,q)
    Return: rmse between forecasted data and actual test dataset
    """
    # data splitting
    trainset = data.iloc[:-self.stp, :]
    validation = data.iloc[-self.stp:, :]

    # data training
    try:
      bestModel = ARIMA(trainset, order=pdq, freq=self.freq).fit()
      # data forecasting
      fc_result, se, conf = bestModel.forecast(steps=self.stp)
    except:
      return None

    # calculate mean square error
    rmse = np.mean((fc_result - validation.values)**2)**.5

    return rmse

  def GSparams(self, data, d):
    """
    Grid Search for best parameters
    """
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #________________i WILL USE PARALLEL PROGRAMMING SOLVE___________
    pool = mp.Pool(mp.cpu_count())

    pnq = range(1, self.gridSearch)

    pdq_list = list()
    for p in pnq:
      for q in pnq:
        pdq_list.append((p,d,q))

    rmse_result = pool.starmap(self.eval_arima, [(data, pdq) for pdq in pdq_list])
    pool.close()

    best_score, bestParam = float("inf"), None

    rmse_idx = 0
    for rmse in rmse_result:
      if rmse is not None:
        if rmse < best_score:
          best_score = rmse
          bestParam = pdq_list[rmse_idx]
          print("ARIMA%s RMSE=%.3f" %(pdq_list[rmse_idx], rmse))
      rmse_idx += 1

    print("Best ARIMA%s RMSE=%.3f" %(bestParam, best_score))
    #++++++++++++++++++++time is slow++++++it is depreciated+++++++++++++++++++++++++++++++
    # pnq = range(1, self.gridSearch)
    # best_score, bestParam = float("inf"), None
    # for p in pnq:
    #   for q in pnq:
    #     pdq = (p,d,q)
    #     try:
    #       rmse = self.eval_arima(data, pdq)
    #       if rmse < best_score:
    #         best_score, bestParam = rmse, pdq
    #       print("ARIMA%s RMSE=%.3f" %(pdq, rmse))
    #     except:
    #       continue
    # print("Best ARIMA%s RMSE=%.3f" %(bestParam, best_score))
    return bestParam

  def dValue(self, data, pValue=0.05):
    """
    Use adfuller to find the d value
    """
    # make series stationary
    # calculate the p-value for adf test
    adfTest = adfuller(data[self._col])
    # p-value
    adfPValue = adfTest[1]
    # compare it to the confidence level 0.05
    # <0.05, d=0
    dt_diff = data.copy()
    # determine d value
    d = 0
    while adfPValue >= pValue:
      dt_diff = dt_diff[self._col].diff().dropna().reset_index() # need to modify here
      # cal p-value again
      adfTest = adfuller(dt_diff[self._col])
      adfPValue = adfTest[1]
      # diff once, plus one for d
      d += 1
    
    return d

  def forecast(self, data, params, steps=1):
    """
    Use the best parameters to train the model and forecast
    """
    # Best ARIMA Model and Predict one more step
    BEST = ARIMA(data.iloc[:-steps,:], order=params).fit()
    forec, other1, other2 = BEST.forecast(steps=steps)
    fc = forec[-steps:]
    return fc
  
  def forecast_accuracy(self, pred, actual):
    metrics = np.mean((pred - actual)**2)**.5
    return metrics

  def main(self):
    timeSeries = self.data

    totRows = len(timeSeries.index)
    runLoop = totRows - self.window
    # prompt how many rows are there
    print("There are %d total periods" %(totRows))
    print("It will run %d loops" %(runLoop))

    forecast_result = list()
    for i in range(self.window, totRows, 1):
      training = timeSeries.iloc[:i,:]
      # testing = timeSeries.iloc[i:,:]

      d = self.dValue(training, pValue=0.05)

      # find the best p, d, q combination
      # Grid-Search ARIMA Hyperparameters 
      start = time.time()
      best = self.GSparams(training , d)
      end = time.time()
      print("Processing time: %d" %(end-start))

      # Best ARIMA Model and Predict
      fc = self.forecast(training, best, 1)
      forecast_result.append(fc[-1])

    # calculate metrics for the model
    metrics = self.forecast_accuracy(forecast_result, timeSeries.iloc[self.window:,:][self._col].values)
    print(metrics)

    return forecast_result

if __name__ == "__main__":
  RCforecast()