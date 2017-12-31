import pandas as pd 
import numpy as np 
from matplotlib import pylab as plt
from matplotlib import pyplot as plt2
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
from math import *
warnings.filterwarnings("ignore")

dateparse = lambda dates: pd.datetime.strptime(dates, '%M-%Y')
data = pd.read_csv('monthlyRainfall.csv', parse_dates = ['YEAR'], index_col = ['YEAR'], date_parser = dateparse)
dat = pd.date_range(start = '1/1/1971', end = '12/1/2013', freq = "MS")
L = []
for i in data.RAIN:
	L += [i]
#print data.head()
ts = pd.Series(L[840:], index = dat)
print ts

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

def variation(results):
	L = []
	vartn = []
	mean = []
	cov = []
	for i in range(12):
		L += [[]]
		for j in range(i, len(results), 12):
			L[i] += [results[j]]
		vartn += [np.var(L[i])]
		mean += [np.mean(L[i])]
		cov += [100*sqrt(vartn[i])/mean[i]]	
	return np.mean(cov)
	

def monsoonVariation(results):
	L = []
	for i in range(6, len(results), 12):
		mRain = 0
		for  j in range(i, i+4):
			mRain += results[j]
		L += [mRain]
	return 100*sqrt(np.var(L))/np.mean(L)


def annualVariation(results):
	L = []
	for i in range(0, len(results), 12):
		mRain = 0
		for  j in range(i, i+12):
			mRain += results[j]
		L += [mRain]
	return 100*sqrt(np.var(L))/np.mean(L)

test_stationarity(ts)

plt.savefig("MonthlyRainfall.png")
plt.close()


model = ARIMA(ts, order = (8, 0, 1))
results_AR = model.fit(disp = -1)
plt.plot(ts, Label = "Monthly Rainfall Data")
results = results_AR.fittedvalues
results = results.apply(nonZ)
print results
plt.plot(results, color = "RED", Label = "Predicted Monthly Rainfall")
plt.title("RS: %.4f" %(sum((ts - results)**2)))
plt.savefig("MonthlyRainfall_ArimaModel.png")
#p_values = range(9)
#q_values = range(9)
p_values = [0, 12, 24, 36]
q_values = [0, 12, 24, 36]
d_values = [0]
print variation(results)
print monsoonVariation(results)
print annualVariation(results)
"""
error = evaluate_models(ts, p_values, d_values, q_values)
print error
f = open("errorsSeasonal.txt", "w")
f.write(str(p_values))
f.write(str(q_values))
f.write(str(error))
f.close()
"""

















