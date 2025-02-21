# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 19:32:42 2025

@author: geont
"""
import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
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

team_number = 11

data = pd.read_csv('train.csv') # read csv file

yV = np.array(data.iloc[:, team_number + 1]) / 10**6 # normalized time series

# Plot the original Timeseries
plt.figure()
plt.plot(yV)
plt.xlabel("days")
plt.ylabel("total daily incoming solar energy (MJ/m^2)")
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
plt.ylabel("total daily incoming solar energy (MJ/m^2)")
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
'''
# AR(p) Model, Estmation of order p, Fit of AR(p) Model
partial_auto_corr = get_pacf(x, lags = max_lag)

summary_ar1, fittedvalues_ar1, resid_ar1, model_ar1, aic_ar1 = fit_arima_model(
    x, 1, 0, show = True) # AR(1)

# MA(q) Model, Estmation of order q, Fit of MA(q) Model
max_q = 4 # maximum order q tested

for q in np.arange(1, max_q + 1):
    summary_ma1, fittedvalues_ma1, resid_ma1, model_ma1, aic_ma1 = fit_arima_model(
        x, 0, q, show = True) # MA(q)
'''
# ARMA(p,q) Model
warnings.filterwarnings("error")

max_p = 4
max_q = 4
aic_arma = np.full(shape = (max_p + 1, max_q + 1), fill_value = np.nan)
ljung_pval = np.full(shape = (max_p + 1, max_q + 1), fill_value = np.nan)

for p in np.arange(max_p + 1):
    for q in np.arange(max_q + 1):
        if p != 0 or q != 0:
            print(f"ARMA({p},{q})---------------------")
            try:
                summary, fittedvalues, resid, model, aic_arma[p, q] = fit_arima_model(
                    x, p, q, show = False) # ARMA(p,q)
                ljung_val, ljung_pval[p, q] = portmanteau_test(resid, max_lag)
            except:
                print("Either non-stationary AR part or non-invertable MA part")
            
warnings.resetwarnings()

plt.figure()
plt.matshow(aic_arma)
plt.colorbar()
plt.xticks(np.arange(max_q + 1), [str(i) for i in np.arange(max_q + 1)])
plt.yticks(np.arange(max_p + 1), [str(i) for i in np.arange(max_p + 1)])
plt.xlabel("q")
plt.ylabel("p")
plt.title("AIC - ARMA(p,q)")

plt.figure()
plt.matshow(ljung_pval)
plt.colorbar()
plt.xticks(np.arange(max_q + 1), [str(i) for i in np.arange(max_q + 1)])
plt.yticks(np.arange(max_p + 1), [str(i) for i in np.arange(max_p + 1)])
plt.xlabel("q")
plt.ylabel("p")
plt.title("p-value (Portmanteau Test for residuals) - ARMA(p,q)")

'''
summary, fittedvalues, resid, model, aic_arma = fit_arima_model(
    x, 2, 3, show = True) # ARMA(2, 3)
'''