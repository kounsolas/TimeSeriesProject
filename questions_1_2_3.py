# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pmd
from statsmodels.tsa.arima.model import ARIMA

def seasonal_components(xV, period):
    '''
    computes the periodic time series comprised of repetetive
    patterns of seasonal components given a time series and the season
    (period).
    '''
    n = xV.shape[0]
    sV = np.full(shape=(n,), fill_value=np.nan)
    monV = np.full(shape=(period,), fill_value=np.nan)
    for i in np.arange(period):
        monV[i] = np.mean(xV[i:n:period])
    monV = monV - np.mean(monV)
    for i in np.arange(period):
        sV[i:n:period] = monV[i] * np.ones(shape=len(np.arange(i, n, period)))
    return sV

def portmanteau_test(xV, maxtau, show=False):
    '''
    PORTMANTEAULB hypothesis test (H0) for independence of time series:
    tests jointly that several autocorrelations are zero.
    It computes the Ljung-Box statistic of the modified sum of
    autocorrelations up to a maximum lag, for maximum lags
    1,2,...,maxtau.
    '''
    ljung_result = acorr_ljungbox(xV, lags=maxtau)

    # Extract the 'lb_stat' and 'lb_pvalue' columns from the DataFrame
    ljung_val = ljung_result['lb_stat'].values
    ljung_pval = ljung_result['lb_pvalue'].values

    if show:
        fig, ax = plt.subplots(1, 1)
        ax.scatter(np.arange(1, len(ljung_pval) + 1), ljung_pval)
        ax.axhline(0.05, linestyle='--', color='r')
        ax.set_title('Ljung-Box Portmanteau test')
        ax.set_yticks(np.arange(0, 1.1))
        plt.show()

    return ljung_val, ljung_pval

def get_acf(xV, lags=10, alpha=0.05, show=True):
    '''
    calculate acf of timeseries xV to lag (lags) and show
    figure with confidence interval with (alpha)
    '''
    acfV = acf(xV, nlags=lags)[1:]
    z_inv = norm.ppf(1 - alpha / 2)
    upperbound95 = z_inv / np.sqrt(xV.shape[0])
    lowerbound95 = -upperbound95
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(np.arange(1, lags + 1), acfV, marker='o')
        ax.axhline(upperbound95, linestyle='--', color='red', label=f'Conf. Int {(1 - alpha) * 100}%')
        ax.axhline(lowerbound95, linestyle='--', color='red')
        ax.set_title('Autocorrelation')
        ax.set_xlabel('Lag')
        ax.set_xticks(np.arange(1, lags + 1))
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.15)
        ax.legend()
    return acfV

def get_pacf(xV, lags=10, alpha=0.05, show=True):
    '''
    calculate pacf of timeseries xV to lag (lags) and show
    figure with confidence interval with (alpha)
    '''
    pacfV = pacf(xV, nlags=lags, method='ols-adjusted')[1:]
    z_inv = norm.ppf(1 - alpha / 2)
    upperbound95 = z_inv / np.sqrt(xV.shape[0])
    lowerbound95 = -upperbound95
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(np.arange(1, lags + 1), pacfV, marker='o')
        ax.axhline(upperbound95, linestyle='--', color='red', label=f'Conf. Int {(1 - alpha) * 100}%')
        ax.axhline(lowerbound95, linestyle='--', color='red')
        ax.set_title('Partial Autocorrelation')
        ax.set_xlabel('Lag')
        ax.set_xticks(np.arange(1, lags + 1))
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.15)
        ax.legend()
    return pacfV

def fit_arima_model(xV, p, q, d=0, show=False):
    '''
    fit ARIMA(p, d, q) in xV
    returns: summary (table), fittedvalues, residuals, model, AIC
    '''
    alpha = 0.05
    z_inv = norm.ppf(1 - alpha / 2)

    try:
        model = ARIMA(xV, order=(p, d, q)).fit()
    except:
        return np.nan
    summary = model.summary()
    fittedvalues = model.fittedvalues
    fittedvalues = np.array(fittedvalues).reshape(-1, 1)
    resid = model.resid
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(xV, label='Original', color='blue')
        ax.plot(fittedvalues, label='FittedValues', color='red', linestyle='--', alpha=0.9)
        ax.legend()
        ax.set_title(f'ARIMA({p}, {d}, {q})')
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.scatter(np.arange(len(resid)), resid)
        xlims = plt.xlim([0, len(xV) - 1])
        ax.hlines([-z_inv, z_inv], xlims[0], xlims[1], colors = 'red')
        plt.title(f'Residuals ARIMA({p}, {d}, {q})')
        plt.legend(["Residuals", f"Conf. Int {(1 - alpha) * 100}%"])
    return summary, fittedvalues, resid, model, model.aic

def arimamodel(xV):
    '''
    BUILT-IN SOLUTION FOR DETECTING BEST ARIMA MODEL MINIMIZING AIC
    https://alkaline-ml.com/pmdarima/index.html
    '''
    autoarima_model = pmd.auto_arima(xV,
                                     start_p=1, start_q=1,
                                     max_p=5, max_q=5, d=0,
                                     test="adf", stepwise=False,
                                     trace=True, information_criterion='aic')
    return autoarima_model

team_number = 11

data = pd.read_csv('train.csv') # read csv file

yV = np.array(data.iloc[:, team_number + 1])

# Plot the original Timeseries
plt.figure()
plt.plot(yV)
plt.xlabel("days")
plt.ylabel("total daily incoming solar energy (J/m^2)")
plt.title("Original Timeseries")
plt.show()

dy = np.diff(yV) # first differences of y
i_zero = [i for i,x in enumerate(dy) if x == 0] # indices where first differences are zero

y = yV
N = len(y)
for i in i_zero:
    # occurence in first year
    if i < 365:
        y[i + 1] = yV[i + 365]
    # occurence in last year
    elif i >= N - 365:
        y[i + 1] = yV[i - 365]
    # occurence in indermediate years
    else:
        y[i + 1] = 0.5 * (yV[i - 365] + yV[i + 365])


### 1. Remove seasonality

period = 365 # seasonality
s = seasonal_components(y, period) # seasonal component
x = y - s # stationary timeseries

# Plot the stationary timeseries
plt.figure()
plt.plot(x)
plt.xlabel("days")
plt.ylabel("total daily incoming solar energy (J/m^2)")
plt.title("Stationary A Timeseries (without seasonality)")
plt.show()

## 2. White Noise Hypothesis Testing (H0: ρτ = 0)
max_lag = 20
alpha = 0.05

# Plot autocorrelation function
auto_corr = get_acf(x, lags = max_lag)

# Portmanteau test
ljung_val, ljung_pval = portmanteau_test(x, max_lag, show=True)
print("2. White Noise Hypothesis Testing (H0: ρτ = 0)")
if ljung_pval[max_lag-1] < alpha:
    print(f"p-value = {ljung_pval[max_lag-1]:.5e}: The stationay A timeseries is not white noise")
else:
    print(f"p-value = {ljung_pval[max_lag-1]:.5e}: The stationay A timeseries is white noise")

## 3. Search for Best Linear Model

best_linear_model = arimamodel(x)
print(best_linear_model.summary())

best_p = best_linear_model.order[0]
best_q = best_linear_model.order[2]
summary, fittedvalues, resid, model, aic = fit_arima_model(x, p = best_p, q = best_q, d=0, show=False)

# Check if the residuals are white noise
print(f"Mean of the residuals: {np.mean(resid):.2f}")
print(f"Variance of the residuals: {np.var(resid):.2f}")
acfV = get_acf(resid, lags=20, alpha=0.05, show=True)

# Portmanteau test for Residuals
maxtau = 20
ljung_val, ljung_pval = portmanteau_test(resid, maxtau=maxtau, show=True)
if np.any(ljung_pval[maxtau - 1] < 0.05):
    print(f"p-value = {ljung_pval[maxtau-1]:.5e}: Reject the null hypothesis that the residuals are independently distributed.")
else:
    print(f"p-value = {ljung_pval[maxtau-1]:.5e}: Accept the null hypothesis that the residuals are independently distributed.")