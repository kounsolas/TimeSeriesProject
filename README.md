# Time Series Project

Time series analysis and forecasting of daily solar energy for the Time Series course at Aristotle University of Thessaloniki. `train.csv` holds daily solar energy readings from 98 Oklahoma Mesonet stations (1994–2007). The project analyses one station (team 11, `BREC`).

## Questions

| # | Task | Method |
|---|---|---|
| 1 | Remove seasonality (period 365) | Seasonal components |
| 2 | Test whether the result is white noise | ACF, Ljung–Box portmanteau test |
| 3 | Find the best linear model | AR, MA and ARMA fits; AR(1) is selected |
| 4–6 | Forecast with the best linear model and with the seasonal mean | Multi-step prediction, NRMSE |
| 7 | Non-linear models | Delay embedding (mutual information, false nearest neighbours), local average and local linear predictors, MLP, SVR |
| 8 | Pick and evaluate the best non-linear model | NRMSE, residual analysis, correlation dimension |

## Files

- `project.ipynb`: the full analysis with explanations and plots
- `questions_1_2_3.py`, `questions_4_5_6.py`, `question_7.py`, `question_8.py`: the same work as standalone scripts
- `train.csv`: the dataset

## Run

```bash
pip install numpy pandas matplotlib scipy statsmodels pmdarima scikit-learn nolds nolitsa jupyter
jupyter notebook project.ipynb
```

`nolitsa` is not on PyPI. Install it from GitHub with `pip install git+https://github.com/manu-mannattil/nolitsa.git`.
