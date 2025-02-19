# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:15:25 2025

@author: geont
"""

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from matplotlib import pyplot as plt
import dimension
import delay

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

def falsenearestneighbors(xV, m_max=10, tau=1, show=False):
    dim = np.arange(1, m_max + 1)
    f1, _, _ = dimension.fnn(xV, tau=tau, dim=dim, window=10, metric='cityblock', parallel=False)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.scatter(dim, f1)
        ax.axhline(0.01, linestyle='--', color='red', label='1% threshold')
        ax.set_xlabel('m')
        ax.set_title(f'FNN ({m_max}), tau = {tau}')
        ax.set_xticks(dim)
        ax.legend()
        plt.show()
    return f1

def get_nrmse(target, predicted):
    se = (target - predicted) ** 2
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    return rmse / np.std(target)

def localfitnrmse(xV, tau, m, Tmax, nnei, q, show=''):
    '''
     LOCALFITNRMSE makes fitting using a local model of zeroth order (average
    % mapping or nearest neighbor mappings if only one neighbor is chosen) or a
    % local linear model and computes the fitting error for T-step ahead. For
    % the search for neighboring points it uses the Matlab k-d-tree search.
    % The fitting here means that predictions are made for all the points in
    % the data set (in-sample prediction). The prediction error statistic
    % (NRMSE measure) for the T-step ahead predictions is the goodness-of-fit
    % statistic.
    % The state space reconstruction is done with the method of delays having
    % as parameters the embedding dimension 'm' and the delay time 'tau'.
    % The local prediction model is one of the following:
    % Ordinary Least Squares, OLS (standard local linear model): if the
    % truncation parameter q >= m
    % Principal Component Regression, PCR, project the parameter space of the
    % model to only q of the m principal axes: if 0<q<m
    % Local Average Mapping, LAM: if q=0.
    % The local region is determined by the number of neighbours 'nnei'.
    % The k-d-tree data structure is utilized to speed up computation time in
    % the search of neighboring points and the implementation of Matlab is
    % used.
    % INPUTS:
    %  xV      : vector of the scalar time series
    %  tau     : the delay time (usually set to 1).
    %  m       : the embedding dimension.
    %  Tmax    : the prediction horizon, the fit is made for T=1...Tmax steps
    %            ahead.
    %  nnei    : number of nearest neighbors to be used in the local model.
    %            If k=1,the nearest neighbor mapping is the fitted value.
    %            If k>1, the model as defined by the input patameter 'q' is
    %            used.
    %  q       : the truncation parameter for a normalization of the local
    %            linear model if specified (to project the parameter space of
    %            the model, using Principal Component Regression, PCR, locally).
    %            if q>=m -> Ordinary Least Squares, OLS (standard local linear
    %                       model, no projection)
    %            if 0<q<m -> PCR(q)
    %            if q=0 -> local average model (if in addition nnei=1 ->
    %            then the zeroth order model is applied)
    %  tittxt  : string to be displayed in the title of the figure
    %            if not specified, no plot is made
    % OUTPUT:
    %  nrmseV  : vector of length Tmax, the nrmse of the fit for T-mappings,
    %            T=1...Tmax.
    %  preM    : the matrix of size nvec x (1+Tmax) having the fit (in-sample
    %            predictions) for T=1,...,Tmax, for each of the nvec
    %            reconstructed points from the whole time series. The first
    %            column has the time of the target point and the rest Tmax
    %            columns the fits for T=1,...,Tmax time steps ahead.
    '''
    if q > m:
        q = int(m)
    n = xV.shape[0]

    if n < 2 * (m - 1) * tau - Tmax:
        print('too short timeseries')
        return

    nvec = n - (m - 1) * tau - Tmax
    xM = np.full(shape=(nvec, m), fill_value=np.nan)

    for j in np.arange(m):
        xM[:, [m - j - 1]] = xV[j * tau:nvec + j * tau].reshape(-1,1)
    from scipy.spatial import KDTree
    kdtreeS = KDTree(xM)
    preM = np.full(shape=(nvec, Tmax), fill_value=np.nan)
    _, nneiindM = kdtreeS.query(xM, k=nnei + 1, p=2)
    nneiindM = nneiindM[:, 1:]
    for i in np.arange(nvec):
        neiM = xM[nneiindM[i]]
        yV = xV[nneiindM[i] + m * tau]
        if q == 0 or nnei == 1:
            preM[i, 0] = np.mean(yV)
        else:
            mneiV = np.mean(neiM, axis=0)
            my = np.mean(yV)
            zM = neiM - mneiV
            [Ux, Sx, Vx] = np.linalg.svd(zM, full_matrices=False)
            Sx = np.diag(Sx)
            Vx = Vx.T
            tmpM = Vx[:, :q] @ (np.linalg.inv(Sx[:q, :q]) @ Ux[:, :q].T)
            lsbV = tmpM @ (yV - my)
            preM[i] = my + (xM[i,] - mneiV) @ lsbV
    if Tmax > 1:
        winnowM = np.full(shape=(nvec, (m - 1) * tau + 1), fill_value=np.nan)
        for i in np.arange((m - 1) * tau + 1):
            winnowM[:, [i]] = xV[i:nvec + i].reshape(-1,1)
        for T in np.arange(2, Tmax + 1):
            winnowM = np.concatenate([winnowM, preM[:, [T - 2]]], axis=1)
            targM = winnowM[:, -1:-(m * tau + 1):-tau]
            _, nneiindM = kdtreeS.query(targM, k=nnei, p=2)

            for i in np.arange(nvec):
                neiM = xM[nneiindM[i], :]
                yV = xV[nneiindM[i] + (m - 1) * tau + 1]
                if q == 0 or nnei == 1:
                    preM[i, T - 1] = np.mean(yV)
                else:
                    mneiV = np.mean(neiM, axis=0)
                    my = np.mean(yV)
                    zM = neiM - mneiV
                    [Ux, Sx, Vx] = np.linalg.svd(zM, full_matrices=False)
                    Sx = np.diag(Sx)
                    Vx = Vx.T
                    tmpM = Vx[:, :q] @ (np.linalg.inv(Sx[:q, :q]) @ Ux[:, :q].T)
                    lsbV = tmpM @ (yV - my)
                    preM[i, T - 1] = my + (targM[i, :] - mneiV) @ lsbV

    nrmseV = np.full(shape=(Tmax, 1), fill_value=np.nan)
    idx = (np.arange(nvec) + (m - 1) * tau).astype(int)
    for t_idx in np.arange(1, Tmax + 1):
        nrmseV[t_idx - 1] = get_nrmse(target=xV[idx + t_idx,], predicted=preM[:, [t_idx - 1]])
    if show == 'True':
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(1, Tmax + 1), nrmseV, marker='x')
        ax.set_xlabel('prediction time T')
        ax.set_ylabel('NRMSE(T)')
        if q == 0:
            ax.set_title("Local Average Predictor")
        elif q >= m:
            ax.set_title("Standard Local Linear Model")
        elif q > 0: 
            ax.set_title("Local Linear Model with PCR")
    return nrmseV, preM

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

team_number = 11

data = pd.read_csv('train.csv') # read csv file

yV = np.array(data.iloc[:, team_number + 1]) / 10**6 # normalized time series

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

### Estimation of Delay Time (tau) 

mi = delay.dmi(x, maxtau = 10)
plt.plot(mi, '.')
plt.xticks(range(10))
plt.title("Mutual Information Xt, Xt+τ")
plt.xlabel("τ")
plt.ylabel("I(τ)")

### Estimation of Embedding Dimension (m)

f1 = falsenearestneighbors(x, tau = 7, show = True)

### Local Average Prediction

best_tau = 3
best_m = 5
T = 5
max_lag = 10
alpha = 0.05

nrmse_av, pred_av = localfitnrmse(x, tau = best_tau, m = best_m, Tmax = T, nnei = 50, q = 0, show = 'False')

e_av = x[(best_m - 1) * best_tau + T:] - pred_av[:, 0] # error for one step prediction
ljung_val_av, ljung_pval_av = portmanteau_test(e_av, max_lag) 
if ljung_pval_av < alpha:
    print(f"Local Average Prediction p-value = {ljung_pval_av:.5e}: The residuals are not white noise")
else:
    print(f"Local Average Prediction p-value = {ljung_pval_av:.5e}: The residuals are white noise")

plt.figure()
plt.plot(x)
plt.plot(np.arange((best_m - 1) * best_tau + T, N), pred_av[:, 0])
plt.title("Local Average Predictor")
plt.legend(["real timeseries","prediction (1 step ahead)"])
plt.xlabel("t")

### Standard Local Linear Model (OLS)

nrmse_ols, pred_ols = localfitnrmse(x, tau = best_tau, m = best_m, Tmax = 5, nnei = 50, q = best_m, show = 'False')

e_ols = x[(best_m - 1) * best_tau + T:] - pred_ols[:, 0] # error for one step prediction
ljung_val_ols, ljung_pval_ols = portmanteau_test(e_ols, max_lag) 
if ljung_pval_ols < alpha:
    print(f"Standard Local Linear Model p-value = {ljung_pval_ols:.5e}: The residuals are not white noise")
else:
    print(f"Standard Local Linear Model p-value = {ljung_pval_ols:.5e}: The residuals are white noise")

plt.figure()
plt.plot(x)
plt.plot(np.arange((best_m - 1) * best_tau + T, N), pred_ols[:, 0])
plt.title("Standard Local Linear Model")
plt.legend(["real timeseries","prediction (1 step ahead)"])
plt.xlabel("t")

### Local Linear Model with PCR

proj_dim = best_m - 1
nrmse_pcr, pred_pcr = localfitnrmse(x, tau = best_tau, m = best_m, Tmax = 5, nnei = 50, q = proj_dim, show = 'False')

e_pcr = x[(best_m - 1) * best_tau + T:] - pred_pcr[:, 0] # error for one step prediction
ljung_val_pcr, ljung_pval_pcr = portmanteau_test(e_pcr, max_lag) 
if ljung_pval_pcr < alpha:
    print(f"Local Linear Model with PCR p-value = {ljung_pval_pcr:.5e}: The residuals are not white noise")
else:
    print(f"Local Linear Model with PCR p-value = {ljung_pval_pcr:.5e}: The residuals are white noise")

plt.figure()
plt.plot(x)
plt.plot(np.arange((best_m - 1) * best_tau + T, N), pred_pcr[:, 0])
plt.title("Local Linear Model with PCR")
plt.legend(["real timeseries","prediction (1 step ahead)"])
plt.xlabel("t")