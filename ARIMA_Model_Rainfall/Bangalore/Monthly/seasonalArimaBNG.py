import pandas as pd 
import numpy as np 
from matplotlib import pylab as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import warnings
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
np.random.seed(1)


warnings.filterwarnings("ignore")

#dateparse = lambda dates: pd.datetime.strptime(dates, '%M-%Y')
data = pd.read_csv('monthlyBanRain.csv')
dat = pd.date_range(start = '1/1/1971', end = '12/1/2002', freq = "MS")
L = []
for i in data.Rain:
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


res = sm.tsa.seasonal_decompose(ts)
seas = res.seasonal
trend = res.trend
irr = res.resid
plt.plot(seas, label = "Seasonal")
plt.plot(irr, color = "BLACK", label = "Irregular")
plt.plot(trend, color = "RED", label = "Trend")
plt.legend(loc = "best")
plt.savefig("Components")
test_stationarity(seas)
test_stationarity(trend)
#test_stationarity(irr)