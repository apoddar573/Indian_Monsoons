import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
from datetime import datetime
import warnings
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import *
warnings.filterwarnings("ignore")

data = pd.read_csv("annualRain.csv")
dates = pd.date_range(start = '1/1/1901', end = '1/1/2002', freq = 'AS')
print data.Rain
print dates
L = []
for i in range(102):
	L += [data.Rain[i]]
ts = pd.Series(L, index = dates)
print ts

def test_stationarity(timeseries):
	#determining rolling statistics
	rolmean = pd.rolling_mean(timeseries, window = 12)
	rolstd = pd.rolling_std(timeseries, window = 12)

	#plotting rolling data
	orig = plt.plot(timeseries, color = 'blue', label = 'Original')
	mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
	mean = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
	plt.title('Rolling Mean and Standard Deviation')
	plt.legend(loc = 'best')

	#Dickey - Fuller Test
	dftest =  adfuller(timeseries, autolag = 'AIC')
	dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', 'Lags Used', 'Number of observations used'])
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
	best_score = float("inf")
	best_cfg = None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p, d, q)
				try:
					mse = evaluate_arima_model(datasets, order)
					if mse<best_score:
						best_score, best_cfg = mse, order
					print("ARIMA%s MSE:%.3f" % (order, mse))
				except:
					continue
	print("Best ARIMA%s MSE:%.3f" % (best_cfg, best_score))

def nonZ(x):
	if x<0:
		return 0
	else:
		return x

test_stationarity(ts)

plt.savefig("AnnualRainfallKamrup.png")
plt.close()

model = ARIMA(ts, order = (0, 0, 0))
results_AR = model.fit(disp = -1)
plt.plot(ts, Label = "Annual Rainfall Data")
results = results_AR.fittedvalues.apply(nonZ)
plt.plot(results, color = "RED", Label = "Predicted Annual Rainfall")
plt.title("RS: %.4f" %(sqrt(np.mean((ts - results)**2))))
plt.savefig("AnnualRainfallKMP_ArimaModel.png")
m =  np.mean(results.values)
sd =  np.var(results.values)
print("The variantion is %.3f %%" % (100*sqrt(sd)/m))

p_values = [0 , 1, 2, 3, 4, 5, 6]
q_values = [0 , 1, 2, 3, 4]
d_values = [0]
#evaluate_models(ts, p_values, d_values, q_values)
