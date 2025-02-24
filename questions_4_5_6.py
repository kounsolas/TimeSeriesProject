# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from scipy.spatial import KDTree

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

def portmanteau_test(xV, maxtau, show = False):
    '''
    PORTMANTEAULB hypothesis test (H0) for independence of time series:
    tests jointly that several autocorrelations are zero.
    It computes the Ljung-Box statistic of the modified sum of
    autocorrelations up to a maximum lag, for maximum lags
    1,2,...,maxtau.
    '''
    ljung = acorr_ljungbox(xV, lags=maxtau)
    ljung_val, ljung_pval = ljung.iloc[0].iloc[0], ljung.iloc[0].iloc[1]
    '''
    if show:
        fig, ax = plt.subplots(1, 1)
        ax.scatter(np.arange(len(ljung_pval)), ljung_pval)
        ax.axhline(0.05, linestyle='--', color='r')
        ax.set_title('Ljung-Box Portmanteau test')
        ax.set_yticks(np.arange(0, 1.1))
        plt.show()
    '''
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

def calculate_fitting_error(xV, model, Tmax=20, show=False):
    '''
    calculate fitting error with NRMSE for given model in timeseries xV
    till prediction horizon Tmax
    returns:
    nrmseV
    preds: for timesteps T=1, 2, 3
    '''
    nrmseV = np.full(shape=Tmax, fill_value=np.nan)
    nobs = len(xV)
    xV_std = np.std(xV)
    vartar = np.sum((xV - np.mean(xV)) ** 2)
    predM = []
    tmin = np.max(
        [len(model.arparams), len(model.maparams), 1])  # start prediction after getting all lags needed from model
    for T in np.arange(1, Tmax + 1):
        errors = []
        predV = np.full(shape=nobs, fill_value=np.nan)
        for t in np.arange(tmin, nobs - T):
            pred_ = model.predict(start=t, end=t + T - 1)
            # predV.append(pred_[-1])
            ytrue = xV[t + T - 1]
            predV[t + T - 1] = pred_[-1]
            error = pred_[-1] - ytrue
            errors.append(error)
        predM.append(predV)
        errors = np.array(errors)
        mse = np.mean(np.power(errors, 2))
        rmse = np.sqrt(mse)
        nrmseV[T - 1] = (rmse / xV_std)
        # nrmseV[T] = (np.sum(errors**2) / vartar)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(np.arange(1, Tmax + 1), nrmseV[1:], marker='x', label='NRMSE');
        ax.axhline(1, color='red', linestyle='--');
        ax.set_title('Fitting Error')
        ax.legend()
        ax.set_xlabel('T')
        ax.set_xticks(np.arange(1, Tmax))
        plt.show()
        # #plot multistep prediction for T=1, 2, 3
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(xV, label='original')
        colors = ['red', 'green', 'black']
        for i, preds in enumerate(predM[:3]):
            ax.plot(preds, color=colors[i], linestyle='--', label=f'T={i + 1}', alpha=0.7)
        ax.legend(loc='best')
        plt.show()
    return nrmseV, predM

team_number = 11

data = pd.read_csv('train.csv') # read csv file

yV = np.array(data.iloc[:, team_number + 1])

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


### Remove seasonality

period = 365 # seasonality    
s = seasonal_components(y, period) # seasonal component
x = y - s # stationary timeseries

### 4. Prediction with Best Linear Model
Tmax = 30
len_test = 365

x_train = x[:-len_test] # training set
x_test = x[-len_test:] # tes set

summary, fittedvalues, resid, model, aic = fit_arima_model(
    x_train, 0, 1, show = False) # ARMA(2, 3)

x_nrmse, x_pred = calculate_fitting_error(x_test, model, Tmax, show = False)
x_pred = np.reshape(x_pred, (len(x_pred), len(x_test)))

y_true = np.tile(y[-len_test:], (Tmax, 1)) 
y_pred = x_pred + np.tile(s[-len_test:], (Tmax, 1))
y_error = y_pred - y_true
y_mse = np.nanmean(np.power(y_error, 2), axis = 1)
y_rmse = np.sqrt(y_mse)
y_nrmse = y_rmse / np.std(y[-len_test:])

# Plot NRMSE with respect to prediction step T
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.plot(np.arange(1, Tmax + 1), y_nrmse, marker='x')
ax.set_xticks(np.arange(1, Tmax + 1))
ax.set_title("NRMSE for Best Linear Model")
ax.set_xlabel("T")
plt.show()


### 5. Prediction with Seasonal Trend
y_train = y[:-len_test] # training set of the non-stationary timeseries
y_test = y[-len_test:] # tes set of the non-stationary timeseries


s_train = seasonal_components(y_train, period)
m_train = np.mean(y_train)

y_pred_seas = s_train[:len(y_test)] + m_train
y_error_seas = y_pred_seas - y_true
y_mse_seas = np.mean(np.power(y_error_seas, 2))
y_rmse_seas = np.sqrt(y_mse_seas)
y_nrmse_seas = y_rmse_seas / np.std(y_test)

print(f"Prediction with Best Linear Model: NRMSE = {y_nrmse[0]:.3f} (T=1)")
print(f"Prediction with Seasonal Trend: NRMSE = {y_nrmse_seas:.3f} (T=1)")

plt.figure()
plt.plot(np.arange(len(y) - len(y_test), len(y)), y_test)
plt.plot(np.arange(len(y) - len(y_test), len(y)), y_pred_seas)
plt.plot(np.arange(len(y) - len(y_test), len(y)), y_pred[0])
plt.xlabel("days")
plt.ylabel("total daily incoming solar energy (J/m^2)")
plt.title("Predictions on Test Set")
plt.legend(["Original Data", "Seasonal Model", "Best Linear Model"])
plt.show()

### 6. SNRMSE for Best Linear Model
y_snrmse = y_nrmse / y_nrmse_seas

# Plot NRMSE and SNRMSE with respect to prediction step T
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.plot(np.arange(1, Tmax + 1), y_nrmse, marker='x')
ax.plot(np.arange(1, Tmax + 1), y_snrmse, marker='o')
ax.axhline(1, color='red', linestyle='--')
ax.set_title('NRMSE and SNRMSE for Best Linear Model')
ax.set_xlabel('T')
ax.set_xticks(np.arange(1, Tmax + 1))
ax.legend(["NRMSE","SNRMSE"])
plt.show()