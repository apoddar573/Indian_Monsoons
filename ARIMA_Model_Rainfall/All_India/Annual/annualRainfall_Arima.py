from math import *
import pandas as pd 
import numpy as np 
from matplotlib import pylab as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y')
data = pd.read_csv('rain.csv', parse_dates = ['YEAR'], index_col = ['YEAR'], date_parser = dateparse)
print data.head()
print data.index
ts = data['#ANN']
print ts['1949']
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


test_stationarity(ts)
plt.savefig("AnnualRainfall.png")
plt.close()

# acf, pacf graphs
lag_acf = acf(ts, nlags= 20)
lag_pacf = pacf(ts, nlags= 20, method = 'ols')

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y = 0, color = 'grey', linestyle = "--")
plt.axhline(y = -1.96/np.sqrt(len(ts)), color = 'gray', linestyle = "--")
plt.axhline(y = 1.96/np.sqrt(len(ts)), color = 'gray', linestyle = "--")
plt.title('Autocorellation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y = 0, color = 'gray', linestyle = "--")
plt.axhline(y = -1.96/np.sqrt(len(ts)), color = 'gray', linestyle = "--")
plt.axhline(y = 1.96/np.sqrt(len(ts)), color = 'gray', linestyle = "--")
plt.title('Partial Autocorellation Function')
plt.savefig("CorellationRainfall.png")
plt.close()

#model = ARIMA(ts_log, order = (0, 2, 2))
model = ARIMA(ts, order = (5, 0, 1))
results_AR = model.fit(disp = -1)
plt.plot(ts, Label = "Actual Rainfall")
results = results_AR.fittedvalues
results = results.apply(nonZ)
plt.plot(results, color = "RED", label = "Predicted Rainfall")
plt.title('ARIMA(5, 0, 1) RS: %.4f'% sum((results_AR.fittedvalues - ts)**2))
plt.legend(loc = "best")
plt.savefig("AnnualRainfall_ArimaModel2.png")
predicted = results_AR.predict(start = '2014-01-01', end = '2015-01-01')
print predicted
forecast = results_AR.forecast(steps = 3)[0]
print forecast
m =  np.mean(results.values)
sd =  np.var(results.values)
print("The variantion is %.3f %%" % (100*sqrt(sd)/m))
