import pandas as pd 
import numpy as np 
from matplotlib import pylab as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import *
import warnings
warnings.filterwarnings("ignore")

dailyrain = []
zeros = []
for i in range(121):
	zeros += [0]
for i in range(2008, 2015):
	name = "dailyBNG" + str(i) + ".csv"
	data = pd.read_csv(name)
	strt = "4/1/" + str(i)
	endt = "11/30/" + str(i)
	dat = pd.date_range(start = strt, end = endt, freq = "D")
	L = []
	for j in data.x:
		L += [j]
	#print data.head()
	if (i+1)%4 == 0:
 		dailyrain = dailyrain + L + zeros + [0]
 	else:
 		dailyrain = dailyrain + L + zeros



dat = pd.date_range(start = "4/1/2008", end = '11/30/2014', freq = "D")
dailyrain = pd.Series(dailyrain[:-121], index = dat)
print dailyrain

def test_stationarity(timeseries):

	#Determine rolling statistics
	rolmean = pd.rolling_mean(timeseries, window = 12)
	rolstd = pd.rolling_std(timeseries, window = 12)


	#plot rolling data
	orig = plt.plot(timeseries, color = 'blue', label = 'Original')
	mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
	std = plt.plot(rolstd, color = 'black', label = 'Rolling std')
	plt.title("Rolling Mean and Standard Deviation")
	plt.legend(loc = 'best')

	#Dickey - Fuller test
	dftest = adfuller(timeseries, autolag = 'AIC')
	dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', 'Lags Used', 'Number of observations Used'])
	for key, value in dftest[4].items():
		dfoutput['Critical Value(%s)'%key] = value
	print dfoutput 

def evaluate_arima_model(X, arima_order):
	train_len = int(len(X)*0.66)
	train, test = X[0:train_len], X[train_len:]
	predictions = list()
	for t in range(len(test)):
		history = X[0: train_len]
		model = ARIMA(history, order = arima_order)
		model_fit = model.fit(disp = -1)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		train_len = train_len+1
	error = mean_squared_error(test, predictions)
	return error

def evaluate_models(datasets, p_values, d_values, q_values):
	L = []
	best_score = float("inf")
	best_cfg = None
	for p in p_values:
		L += [[]]
		for d in d_values:
			for q in q_values:
				L[p] += [float('inf')]
				order = (p, d, q)
				try:
					mse = evaluate_arima_model(datasets, order)
					if mse<best_score:
						best_score, best_cfg = mse, order
					print("ARIMA%s MSE:%.3f" % (order, mse))
					L[p][q] = mse
				except:
					continue
	print("Best ARIMA%s MSE:%.3f" % (best_cfg, best_score))
	return L

def nonZ(x):
	if x<0:
		return 0 
	else:
		return x

test_stationarity(dailyrain)

plt.savefig("DailyBNGRainfall.png")
plt.close()

model = ARIMA(dailyrain, order = (1, 0, 1))
results_AR = model.fit(disp = -1)
plt.plot(dailyrain, Label = "Daily Rainfall Data BNG")
results = results_AR.fittedvalues.apply(nonZ	)
plt.plot(results, color = "RED", Label = "Predicted Daily Rainfall BNG")
plt.title("RS: %.4f" %(sum((dailyrain - results)**2)))
plt.legend(loc = "best")
plt.savefig("DailyBNGRainfall_ArimaModel.png")
p_values = [0 , 1, 2]
q_values = [0 , 1, 2]
d_values = [0]
"""
error = evaluate_models(dailyrain, p_values, d_values, q_values)
print error
f = open("errorsDaily.txt", "w")
f.write(str(error))
f.close()
"""



