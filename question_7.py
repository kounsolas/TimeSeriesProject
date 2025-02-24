# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import norm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from nolitsa import delay, dimension
import nolds

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
        ax.set_title('ACF')
        ax.set_xlabel('Lag')
        ax.set_xticks(np.arange(1, lags + 1))
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.15)
        ax.legend()
    return acfV

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

def embed_data(xV, m=3, tau=1):
    """Time-delay embedding.
    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    m : int
        Embedding dimension (order)
    tau : int
        Delay.
    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(xV)
    nvec = N - (m - 1) * tau
    xM = np.zeros(shape=(nvec, m))
    for i in np.arange(m):
        xM[:, m - i - 1] = xV[i * tau:nvec + i * tau]
    return xM

def nrmse(trueV, predictedV):
    vartrue = np.sum((trueV - np.mean(trueV)) ** 2)
    varpred = np.sum((predictedV - trueV) ** 2)
    return np.sqrt(varpred / vartrue)

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
        xM[:, [m - j - 1]] = xV[j * tau:nvec + j * tau].reshape(-1, 1)
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
        # winnowM = np.full(shape=(nvec, (m - 1) * tau + 1), fill_value=np.nan)
        winnowM = np.full(shape=(nvec, m * tau), fill_value=np.nan)  # Should be (5062, 25)
        # for i in np.arange(m * tau):
        for i in range(min(m * tau, winnowM.shape[1])):  # Ensure i doesn't exceed column count
            winnowM[:, [i]] = xV[i:nvec + i].reshape(-1, 1)
        for T in np.arange(2, Tmax + 1):
            winnowM = np.concatenate([winnowM, preM[:, [T - 2]]], axis=1)
            # targM = winnowM[:, :-(m + 1) * tau:-tau]
            targM = winnowM[:, -5:]  # Select the last 5 columns

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
    if show:
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(1, Tmax + 1), nrmseV, marker='x')
        ax.set_xlabel('prediction time T')
        ax.set_ylabel('NRMSE(T)')
        if q==0:
            ax.set_title(f"Local Average Predictor (m = {m}, τ = {tau})")
        elif q>=m:
            ax.set_title(f"Standard Local Linear Model (m = {m}, τ = {tau})")
        elif q>0:
            ax.set_title(f"Local Linear Model with PCR (m = {m}, τ = {tau})")
    return nrmseV, preM



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

def localpredictnrmse(xV, nlast, m, tau=1, Tmax=1, nnei=1, q=0, show=''):
    xV = xV.reshape(-1, )
    n = xV.shape[0]
    if nlast > n - 2 * m * tau:
        print('test set too large')
    n1 = n - nlast
    if n1 < 2 * (m - 1) * tau - Tmax:
        print('the length of training set is too small for this data size')
    n1vec = n1 - (m - 1) * tau - 1
    xM = np.full(shape=(n1vec, m), fill_value=np.nan)
    for j in np.arange(m):
        xM[:, m - j - 1] = xV[j * tau:n1vec + j * tau]
    from scipy.spatial import KDTree
    kdtreeS = KDTree(xM)

    # For each target point, find neighbors, apply the linear models and keep track
    # of the predicted values each prediction time.
    ntar = nlast - Tmax + 1;
    preM = np.full(shape=(ntar, Tmax), fill_value=np.nan)
    winnowM = np.full(shape=(ntar, (m - 1) * tau + 1), fill_value=np.nan)

    ifirst = n1 - (m - 1) * tau
    for i in np.arange((m - 1) * tau + 1):
        winnowM[:, i] = xV[ifirst + i - 1 : ifirst + ntar + i - 1]

    for T in np.arange(1, Tmax + 1):
        targM = winnowM[:, -m * tau::tau]
        _, nneiindM = kdtreeS.query(targM, k=nnei, p=2)
        for i in np.arange(ntar):
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
        winnowM = np.concatenate([winnowM, preM[:, [T - 1]]], axis=1)
    nrmseV = np.full(shape=(Tmax, 1), fill_value=np.nan)

    start_idx = (n1vec + (m - 1) * tau)
    end_idx = start_idx + preM.shape[0]
    for t_idx in np.arange(1, Tmax + 1):
        nrmseV[t_idx - 1] = nrmse(trueV=xV[start_idx + t_idx:end_idx + t_idx], predictedV=preM[:, t_idx - 1])
    if show=='True':
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(1, Tmax + 1), nrmseV, marker='x')
        ax.set_xlabel('prediction time T')
        ax.set_ylabel('NRMSE(T)')
        ax.axhline(1., color='yellow')
        ax.set_title(f'NRMSE(T), m={m}, tau={tau}, q={q}, n={n}, nlast={nlast}')
    return nrmseV, preM

def correlationdimension(xV, tau, m_max, fac=4, logrmin=-1e6, logrmax=1e6, show=False):
    m_all = np.arange(1, m_max + 1)
    corrdimV = []
    logrM = []
    logCrM = []
    polyM = []

    for m in m_all:
        corrdim, *corrData = nolds.corr_dim(xV, m, debug_data=True)
        corrdimV.append(corrdim)
        logrM.append(corrData[0][0])
        logCrM.append(corrData[0][1])
        polyM.append(corrData[0][2])
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot(m_all, corrdimV, marker='x', linestyle='-.')
        ax.set_xlabel('m')
        ax.set_xticks(m_all)
        ax.set_ylabel('v')
        ax.set_title('Corr Dim vs m')

    return corrdimV, logrM, logCrM, polyM

data_df = pd.read_csv('train.csv')
team_number = 11
team_data = data_df.iloc[:, team_number + 1]

#turn the data into a numpy array
team_data = team_data.to_numpy()

#find the consecutive values
same_val_index = np.where(np.diff(team_data) == 0)
'''
print(f"Value {team_data[same_val_index[0][0]]} at index {same_val_index[0][0]} ")
print(f"Value {team_data[same_val_index[0][0] + 1]} at index {same_val_index[0][0] + 1} ")
print(f"Value {team_data[same_val_index[0][1]]} at index {same_val_index[0][1]} ")
print(f"Value {team_data[same_val_index[0][1] + 1]} at index {same_val_index[0][1] + 1} ")
'''
# Replace the consecutive values with the mean of two values
#check if index - 365 < 0 and index + 365 > len(team_data)

for index in same_val_index[0]:
    if index - 365 < 0:
        # print("First year of data.")
        team_data[index+1] = team_data[index+1+365]
    elif index + 365 > len(team_data):
        # print("Last year of data.")
        team_data[index+1] = team_data[index+1-365]
    else:
        # print("Middle of data.")
        team_data[index+1] = (team_data[index+1-365] + team_data[index+1+365])/2
        
# the data are from 1/1/1994 to 31/12/2007
# i will use the data from 1/1/1994 to 31/12/2006 to train the model
# and the data from 1/1/2007 to 31/12/2007 to test the model

data_df['Date'] = pd.to_datetime(data_df['Date'], format='%Y%m%d')  
data_df.set_index('Date', inplace=True)

split = data_df.loc['1994-01-01':'2006-12-31'].shape[0]
print(f"Number of days in the training data: {split}")

A = np.copy(team_data)
T_season = 365
s = seasonal_components(A, T_season)
A_static= A - s

A_static_train = A_static[:split]
A_static_test = A_static[split:]
s_test = s[split:]
s_train = s[:split]

### 7. Search for Best Nonlinear Model

# Estimation of dealy τ
mi = delay.dmi(A_static,maxtau=10)
plt.figure(figsize=(10,5))
plt.plot(mi)
plt.ylim(top=1)
plt.xlabel('Lag')
plt.ylabel('MI')
plt.title('Mutual Information')
plt.show()

acV = get_acf(A_static, lags=10, alpha=0.05, show=True)
tau_est = 5

# Estimation of embedding dimension m
p_fnn = falsenearestneighbors(A_static, m_max=10, tau=tau_est, show=True)
m_est = 5

# Local Linear Model
Tmax_val=5

# Local Linear Predictor
nrmse_q7a, preM_q7a = localfitnrmse(A_static, tau=tau_est, m=m_est, Tmax=Tmax_val, nnei=50, q=0, show=True) 
resid_q7a = A_static[(m_est - 1) * tau_est + 1 : (m_est - 1) * tau_est + 1 + len(preM_q7a)] - preM_q7a[:, 0]

plt.figure(figsize=(10,5))
plt.plot(A_static, label='original', color='blue')
plt.plot(preM_q7a[:,0],label='One-step ahead prediction', linestyle='--', color='red')
'''
plt.plot(preM_q7[:,1],label='Two-step ahead prediction', linestyle='--', color='green')
plt.plot(preM_q7[:,2],label='Three-step ahead prediction', linestyle='--', color='black')
'''
plt.title("Local Linear Predictor")
plt.legend()
plt.show()

acfV = get_acf(resid_q7a, lags=20, alpha=0.05, show=True)    
mi = delay.dmi(resid_q7a, maxtau=10)
plt.figure(figsize=(10,5))
plt.plot(mi)
plt.ylim(top=1)
plt.xlabel('Lag')
plt.ylabel('MI')
plt.title('Mutual Information for Residuals')
plt.show()

# Standard Local Linear Model
nrmse_q7c, preM_q7c = localfitnrmse(A_static, tau=tau_est, m=m_est, Tmax=5, nnei=50, q=m_est, show=True)
resid_q7c = A_static[(m_est - 1) * tau_est + 1 : (m_est - 1) * tau_est + 1 + len(preM_q7c)] - preM_q7c[:, 0]

plt.figure(figsize=(10,5))
plt.plot(A_static, label='original', color='blue')
plt.plot(preM_q7c[:,0],label='One-step ahead prediction', linestyle='--', color='red')
'''
plt.plot(preM_q7c[:,1],label='Two-step ahead prediction', linestyle='--', color='green')
plt.plot(preM_q7c[:,2],label='Three-step ahead prediction', linestyle='--', color='black')
'''
plt.title("Standard Local Linear Model")
plt.legend()
plt.show()

acf_c = get_acf(resid_q7c, lags=20, alpha=0.05, show=True)
mi = delay.dmi(resid_q7c, maxtau=10)
plt.figure(figsize=(10,5))
plt.plot(mi)
plt.ylim(top=1)
plt.xlabel('Lag')
plt.ylabel('MI')
plt.title('Mutual Information for Residuals')
plt.show()

# Local Linear Model with PCR
q_bar = 2
nrmse_q7b, preM_q7b = localfitnrmse(A_static, tau=tau_est, m=m_est, Tmax=5, nnei=50, q=q_bar, show=True)
resid_q7b = A_static[(m_est - 1) * tau_est + 1 : (m_est - 1) * tau_est + 1 + len(preM_q7b)] - preM_q7b[:, 0]

plt.figure(figsize=(10,5))
plt.plot(A_static, label='original', color='blue')
plt.plot(preM_q7b[:,0],label='One-step ahead prediction', linestyle='--', color='red')
'''
plt.plot(preM_q7b[:,1],label='Two-step ahead prediction', linestyle='--', color='green')
plt.plot(preM_q7b[:,2],label='Three-step ahead prediction', linestyle='--', color='black')
'''
plt.title("Local Linear Model with PCR")
plt.legend()
plt.show()

acf_b = get_acf(resid_q7b, lags=20, alpha=0.05, show=True)
mi = delay.dmi(resid_q7b, maxtau=10)
plt.figure(figsize=(10,5))
plt.plot(mi)
plt.ylim(top=1)
plt.xlabel('Lag')
plt.ylabel('MI')
plt.title('Mutual Information for Residuals')
plt.show()

# Multilayer Perceptron (MLP)
# Embedding
A_static_train_embed = embed_data(A_static_train, m=m_est, tau=tau_est)
A_static_test_embed = embed_data(A_static_test, m=m_est, tau=tau_est)

# Ensure y_train has the same length as X_train
X_train = A_static_train_embed[:-1]
y_train = A_static_train[(m_est - 1) * tau_est:-1] 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = A_static_test_embed[:-1]
y_test = A_static_test[(m_est - 1) * tau_est:-1]
X_test = scaler.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(256, 128,32), activation='relu', momentum=0.8 ,max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

resid_mlp = y_test - y_pred

nrmse_mlp = get_nrmse(y_test, y_pred)
print(f"nrmse_mlp: {nrmse_mlp:.2f}")


acfV = get_acf(resid_mlp, lags=20, alpha=0.05, show=True)
mi = delay.dmi(resid_mlp, maxtau=10)
plt.figure(figsize=(10,5))
plt.plot(mi)
plt.ylim(top=1)
plt.xlabel('Lag')
plt.ylabel('MI')
plt.title('Mutual Information for Residuals')
plt.show()

ljung_val, ljung_pval = portmanteau_test(xV=resid_mlp, maxtau=20, show=True)

# SVM
svr = SVR()
param_grid = {
    'C': [ 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf',  'poly', 'sigmoid'],
    'degree': [2, 3, 5],
    'verbose': [True]
}

grid = GridSearchCV(svr, param_grid, cv=5, n_jobs=-1)

grid.fit(X_train, y_train)

best_svr_embed = grid.best_estimator_
y_pred_svr_embed = best_svr_embed.predict(X_test)

resid_svr_embed = y_test - y_pred_svr_embed
acfV = get_acf(resid_svr_embed, lags=20, alpha=0.05, show=True)

ljung_val, ljung_pval = portmanteau_test(xV=resid_svr_embed, maxtau=20, show=True)

# Convert test data back to its non-static form
A_test_non_static = A_static_test + s_test  

# Adjust predictions
y_pred_non_static = y_pred + s_test[(m_est - 1) * tau_est:-1]

# Plot
plt.figure(figsize=(12,6))
plt.plot(A_test_non_static, label='Original Test Data (Non-Static)', color='blue')
plt.plot(range((m_est - 1) * tau_est + 1, (m_est - 1) * tau_est + 1 + len(y_pred_non_static)), 
         y_pred_non_static, label='MLP Prediction (Non-Static)', color='red', linestyle='--')
plt.legend()
plt.show()

nrmse_embed_mlp = get_nrmse(A_test_non_static[(m_est - 1) * tau_est:(m_est - 1) * tau_est + len(y_pred_non_static)], y_pred_non_static)
nrmse_embed_mlp
