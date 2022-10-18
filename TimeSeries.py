# %% [markdown]
# ## Data preparation

# %% [markdown]
# ### Load libraries

# %%
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt

simplefilter("ignore")  # ignore warnings to clean up output cells

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
from scipy import stats
import openpyxl

from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_pacf

# %% [markdown]
# ### Set graph defaults

# %%
# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
%config InlineBackend.figure_format = 'retina'

# %% [markdown]
# ### Load data

# %%
# Alternate monthly data load - as frequency
data_dir = Path("D:/Python/Input/HondaData")

monthlySales = pd.read_excel(data_dir / "IndependentVarsPy.xlsx",
                           sheet_name = "MonthlySales",
                           parse_dates=["Month"])

monthlySales = monthlySales.rename(columns={'0': 'monthlySales'})

monthlySales = monthlySales.loc[monthlySales['Month'] > '2017-01-01']

monthlySales = monthlySales.set_index("Month").to_period(freq='M')

monthlySales.head()

# %% [markdown]
# #### Load daily (Hijri date) sales data

# %%
# Alternate monthly data load - as frequency
dailyHijriSales = pd.read_excel(data_dir / "IndependentVarsPy.xlsx",
                           sheet_name = "DailyHijriSales",
                           parse_dates=["HijriDate"])

#dailyHijriSales = dailyHijriSales.rename(columns={'0': 'Amount'})

dailyHijriSales = dailyHijriSales.loc[dailyHijriSales['HijriDate'] > '2017-01-01']

dailyHijriSales = dailyHijriSales.set_index("HijriDate").to_period(freq='M')

dailyHijriSales.head()

# %% [markdown]
# ## Time Series on monthly sales data

# %% [markdown]
# ### Trend

# %%
moving_average = monthlySales.rolling(
    window=6,       # 6-month window
    center=True,      # puts the average at the center of the window
    min_periods=3,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

ax = monthlySales.plot(style=".", color="0.5", legend = False)
moving_average.plot(
    ax=ax, linewidth=3, title="Bike Industry and Honda Monthly Whole Sales - 6-Months Moving Average", legend=True,
);

# %%
moving_average = monthlySales.HondaWholeSales.rolling(
    window=6,       # 6-month window
    center=True,      # puts the average at the center of the window
    min_periods=3,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

ax = monthlySales.HondaWholeSales.plot(style=".", color="0.5")
moving_average.plot(
    ax=ax, linewidth=3, title="Honda Monthly Whole Sales - 6-Months Moving Average", legend=False,
);

# %%
from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=monthlySales.index,  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=1,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)
# `in_sample` creates features for the dates given in the `index` argument
X = dp.in_sample()

X.head()

# %%
from sklearn.linear_model import LinearRegression

y = monthlySales["HondaWholeSales"]  # the target

# The intercept is the same as the `const` feature from
# DeterministicProcess. LinearRegression behaves badly with duplicated
# features, so we need to be sure to exclude it here.
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

# %%
print(model.coef_)
print(model.intercept_)

# %%
# Inferential statisticsslope, intercept, r, p, std_err = stats.linregress(X, y)
slope, intercept, r, p, std_err = stats.linregress(X.trend, y)

print("Slope = ", round(slope, 2), ", Intercept = ", round(intercept, 2), ", R value = ", round(r, 2), ", P value = ", round(p, 4), ", Standard Error = ", round(std_err, 2))

# %%
varslope = 64.44
varSE = 8.04

# %%
(varslope*66)+2467.9 

# %%
((varslope + 3*varSE)*66)+2467.9 

# %%
ax = monthlySales.HondaWholeSales.plot(style=".", color="0.5", title="Monthly Whole Sales - Linear Trend")
_ = y_pred.plot(ax=ax, linewidth=3, label="Trend")

# %%
X = dp.out_of_sample(steps=24)

y_fore = pd.Series(model.predict(X), index=X.index)

y_fore.head()

# %%
ax = monthlySales.HondaWholeSales["2017-01":].plot(title="Honda Monthly Whole Sales - 2 years Linear Trend Forecast", **plot_params)
ax = y_pred["2017-01":].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
_ = ax.legend()

# %%
N = 1000
x = np.linspace(0, 10, N)
y = x**2
ones = np.ones(N)

vals = [30, 20, 10] # Values to iterate over and add/subtract from y.

fig, ax = plt.subplots()

for i, val in enumerate(vals):
    alpha = 0.5*(i+1)/len(vals) # Modify the alpha value for each iteration.
    ax.fill_between(x, y+ones*val, y-ones*val, color='red', alpha=alpha)

ax.plot(x, y, color='red') # Plot the original signal

plt.show()

# %% [markdown]
# ### Seasonality

# %%
# annotations: https://stackoverflow.com/a/49238256/5769929
def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax

# %%
X = monthlySales.copy()

# days within a year
X["month"] = X.index.month
X["year"] = X.index.year

fig, ax0 = plt.subplots(1, 1, figsize=(11, 6))
seasonal_plot(X, y="HondaWholeSales", period="year", freq="month", ax=ax0);

# %%
plot_periodogram(monthlySales.HondaSales);

# %%

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq="A", order=8)  # 10 sin/cos pairs for "A"nnual seasonality

dp = DeterministicProcess(
    index=monthlySales.index,
    constant=True,               # dummy feature for bias (y-intercept)
    order=1,                     # trend (order 1 means linear)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (fourier)
    drop=True,                   # drop terms to avoid collinearity
)

X = dp.in_sample()  # create features for dates in tunnel.index

# %%
y = monthlySales["HondaWholeSales"]

model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=40)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(color='0.25', style='.', title="Honda Whole Sales - Seasonal Forecast")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')
_ = ax.legend()

# %% [markdown]
# ### Preparing search trend data

# %%
os.listdir("D:/Python/Input/GoogleTrends/HondaBrandRelatedDaily")

# %%
searchTrendsPath = Path("D:/Python/Input/GoogleTrends/HondaBrandRelatedDaily")

Honda17Q1_17Q2Daily = pd.read_csv(searchTrendsPath / "Honda 17Q1-17Q2 Daily.csv",
                            header=1)
Honda17Q1_17Q2Daily.set_index(
    pd.PeriodIndex(Honda17Q1_17Q2Daily.Day, freq="D"),
    inplace=True)
Honda17Q1_17Q2Daily.drop("Day", axis=1, inplace=True)


Honda17Q3_17Q4Daily = pd.read_csv(searchTrendsPath / "Honda 17Q3-17Q4 Daily.csv",
                            header=1)
Honda17Q3_17Q4Daily.set_index(
    pd.PeriodIndex(Honda17Q3_17Q4Daily.Day, freq="D"),
    inplace=True)
Honda17Q3_17Q4Daily.drop("Day", axis=1, inplace=True)


Honda18Q1_18Q2Daily = pd.read_csv(searchTrendsPath / "Honda 18Q1-18Q2 Daily.csv",
                            header=1)
Honda18Q1_18Q2Daily.set_index(
    pd.PeriodIndex(Honda18Q1_18Q2Daily.Day, freq="D"),
    inplace=True)
Honda18Q1_18Q2Daily.drop("Day", axis=1, inplace=True)


Honda18Q3_18Q4Daily = pd.read_csv(searchTrendsPath / "Honda 18Q3-18Q4 Daily.csv",
                            header=1)
Honda18Q3_18Q4Daily.set_index(
    pd.PeriodIndex(Honda18Q3_18Q4Daily.Day, freq="D"),
    inplace=True)
Honda18Q3_18Q4Daily.drop("Day", axis=1, inplace=True)


Honda19Q1_19Q2Daily = pd.read_csv(searchTrendsPath / "Honda 19Q1-19Q2 Daily.csv",
                            header=1)
Honda19Q1_19Q2Daily.set_index(
    pd.PeriodIndex(Honda19Q1_19Q2Daily.Day, freq="D"),
    inplace=True)
Honda19Q1_19Q2Daily.drop("Day", axis=1, inplace=True)


Honda19Q3_19Q4Daily = pd.read_csv(searchTrendsPath / "Honda 19Q3-19Q4 Daily.csv",
                            header=1)
Honda19Q3_19Q4Daily.set_index(
    pd.PeriodIndex(Honda19Q3_19Q4Daily.Day, freq="D"),
    inplace=True)
Honda19Q3_19Q4Daily.drop("Day", axis=1, inplace=True)


Honda20Q1_20Q2Daily = pd.read_csv(searchTrendsPath / "Honda 20Q1-20Q2 Daily.csv",
                            header=1)
Honda20Q1_20Q2Daily.set_index(
    pd.PeriodIndex(Honda20Q1_20Q2Daily.Day, freq="D"),
    inplace=True)
Honda20Q1_20Q2Daily.drop("Day", axis=1, inplace=True)


Honda20Q3_20Q4Daily = pd.read_csv(searchTrendsPath / "Honda 20Q3-20Q4 Daily.csv",
                            header=1)
Honda20Q3_20Q4Daily.set_index(
    pd.PeriodIndex(Honda20Q3_20Q4Daily.Day, freq="D"),
    inplace=True)
Honda20Q3_20Q4Daily.drop("Day", axis=1, inplace=True)


Honda21Q1_21Q2Daily = pd.read_csv(searchTrendsPath / "Honda 21Q1-21Q2 Daily.csv",
                            header=1)
Honda21Q1_21Q2Daily.set_index(
    pd.PeriodIndex(Honda21Q1_21Q2Daily.Day, freq="D"),
    inplace=True)
Honda21Q1_21Q2Daily.drop("Day", axis=1, inplace=True)


Honda21Q3_21Q4Daily = pd.read_csv(searchTrendsPath / "Honda 21Q3-21Q4 Daily.csv",
                            header=1)
Honda21Q3_21Q4Daily.set_index(
    pd.PeriodIndex(Honda21Q3_21Q4Daily.Day, freq="D"),
    inplace=True)
Honda21Q3_21Q4Daily.drop("Day", axis=1, inplace=True)

Honda22Q1_22Q2Daily = pd.read_csv(searchTrendsPath / "Honda 22Q1-22Q2 Daily.csv",
                            header=1)
Honda22Q1_22Q2Daily.set_index(
    pd.PeriodIndex(Honda22Q1_22Q2Daily.Day, freq="D"),
    inplace=True)
Honda22Q1_22Q2Daily.drop("Day", axis=1, inplace=True)

Honda22Q3_current = pd.read_csv(searchTrendsPath / "Honda 22Q2-9.18 Daily.csv",
                            header=1)
Honda22Q3_current.set_index(
    pd.PeriodIndex(Honda22Q3_current.Day, freq="D"),
    inplace=True)
Honda22Q3_current.drop("Day", axis=1, inplace=True)



# %%
scaleVal1 = (Honda17Q1_17Q2Daily.iloc[-1,0]) / (Honda17Q3_17Q4Daily.iloc[0,0])
Honda17Q3_17Q4DailyScaled = scaleVal1 * Honda17Q3_17Q4Daily.iloc[:,0]
Honda17Q3_17Q4DailyScaled = pd.DataFrame(Honda17Q3_17Q4DailyScaled.round(1))

scaleVal2 = (Honda17Q3_17Q4DailyScaled.iloc[-1,0]) / (Honda18Q1_18Q2Daily.iloc[0,0])
Honda18Q1_18Q2DailyScaled = scaleVal2 * Honda18Q1_18Q2Daily.iloc[:,0]
Honda18Q1_18Q2DailyScaled = pd.DataFrame(Honda18Q1_18Q2DailyScaled.round(1))

scaleVal3 = (Honda18Q1_18Q2DailyScaled.iloc[-1,0]) / (Honda18Q3_18Q4Daily.iloc[0,0])
Honda18Q3_18Q4DailyScaled = scaleVal3 * Honda18Q3_18Q4Daily.iloc[:,0]
Honda18Q3_18Q4DailyScaled = pd.DataFrame(Honda18Q3_18Q4DailyScaled.round(1))

scaleVal4 = (Honda18Q3_18Q4DailyScaled.iloc[-1,0]) / (Honda19Q1_19Q2Daily.iloc[0,0])
Honda19Q1_19Q2DailyScaled = pd.DataFrame(scaleVal4 * Honda19Q1_19Q2Daily.iloc[:,0])
Honda19Q1_19Q2DailyScaled = Honda19Q1_19Q2DailyScaled.round(1)

scaleVal5 = (Honda19Q1_19Q2DailyScaled.iloc[-1,0]) / (Honda19Q3_19Q4Daily.iloc[0,0])
Honda19Q3_19Q4DailyScaled = pd.DataFrame(scaleVal5 * Honda19Q3_19Q4Daily.iloc[:,0])
Honda19Q3_19Q4DailyScaled = Honda19Q3_19Q4DailyScaled.round(1)

scaleVal6 = (Honda19Q3_19Q4DailyScaled.iloc[-1,0]) / (Honda20Q1_20Q2Daily.iloc[0,0])
Honda20Q1_20Q2DailyScaled = pd.DataFrame(scaleVal6 * Honda20Q1_20Q2Daily.iloc[:,0])
Honda20Q1_20Q2DailyScaled = Honda20Q1_20Q2DailyScaled.round(1)

scaleVal7 = (Honda20Q1_20Q2DailyScaled.iloc[-1,0]) / (Honda20Q3_20Q4Daily.iloc[0,0])
Honda20Q3_20Q4DailyScaled = pd.DataFrame(scaleVal7 * Honda20Q3_20Q4Daily.iloc[:,0])
Honda20Q3_20Q4DailyScaled = Honda20Q3_20Q4DailyScaled.round(1)

scaleVal8 = (Honda20Q3_20Q4DailyScaled.iloc[-1,0]) / (Honda21Q1_21Q2Daily.iloc[0,0])
Honda21Q1_21Q2DailyScaled = pd.DataFrame(scaleVal8 * Honda21Q1_21Q2Daily.iloc[:,0])
Honda21Q1_21Q2DailyScaled = Honda21Q1_21Q2DailyScaled.round(1)

scaleVal9 = (Honda21Q1_21Q2DailyScaled.iloc[-1,0]) / (Honda21Q3_21Q4Daily.iloc[0,0])
Honda21Q3_21Q4DailyScaled = pd.DataFrame(scaleVal9 * Honda21Q3_21Q4Daily.iloc[:,0])
Honda21Q3_21Q4DailyScaled = Honda21Q3_21Q4DailyScaled.round(1)

scaleVal10 = (Honda21Q3_21Q4DailyScaled.iloc[-1,0]) / (Honda22Q1_22Q2Daily.iloc[0,0])
Honda22Q1_22Q2DailyScaled = pd.DataFrame(scaleVal10 * Honda22Q1_22Q2Daily.iloc[:,0])
Honda22Q1_22Q2DailyScaled = Honda22Q1_22Q2DailyScaled.round(1)

scaleVal11 = (Honda22Q1_22Q2DailyScaled.iloc[-1,0]) / (Honda22Q3_current.iloc[0,0])
Honda22Q3_currentScaled = pd.DataFrame(scaleVal11 * Honda22Q3_current.iloc[:,0])
Honda22Q3_currentScaled = Honda22Q3_currentScaled.round(1)



# %%
scaledHonda17Q1_CurrentDaily = pd.DataFrame(Honda17Q1_17Q2Daily.iloc[:,0]).append(Honda17Q3_17Q4DailyScaled).append(Honda18Q1_18Q2DailyScaled).append(Honda18Q3_18Q4DailyScaled).append(Honda19Q1_19Q2DailyScaled).append(Honda19Q3_19Q4DailyScaled).append(Honda20Q1_20Q2DailyScaled).append(Honda20Q3_20Q4DailyScaled).append(Honda21Q1_21Q2DailyScaled).append(Honda21Q3_21Q4DailyScaled).append(Honda22Q1_22Q2DailyScaled).append(Honda22Q3_currentScaled).drop_duplicates()

min = scaledHonda17Q1_CurrentDaily.min()
max = scaledHonda17Q1_CurrentDaily.max()

scaled2Honda17Q1_CurrentDaily = ((scaledHonda17Q1_CurrentDaily - min) / (max - min) * 100).round(1)

# %%
outputPath = Path("D:/Python/Output/")

scaled2Honda17Q1_CurrentDaily.to_csv(outputPath / "SearchTrendAnalysis/scaledHonda17Q1ToCurrentDaily.csv")

# %%
#Read google trend data

hondaBike_trends = pd.read_csv("D:/Python/Output/SearchTrendAnalysis/SearchSales17-22.csv")
hondaBike_trends.set_index(
    pd.PeriodIndex(hondaBike_trends.Date, freq="D"),
    inplace=True,
)
hondaBike_trends.drop("Date", axis=1, inplace=True)


# %%

ax = hondaBike_trends["Search"].plot(title='Honda Bike Search Trend', **plot_params)
_ = ax.set(ylabel="Honda Bike Search Trend")

# %%
# Add google search results in Honda daily sales file

ax = hondaBike_trends.plot(
    y=["Search", "Sales"]
)

# %% [markdown]
# #### Weekly data

# %%
#Read google trend data

hondaBike_trends = pd.read_csv("D:/Python/Input/GoogleTrends/Output/WeeklySearchSales 17-22.csv")
hondaBike_trends.set_index(
    pd.PeriodIndex(hondaBike_trends.Week, freq="W"),
    inplace=True,
)
hondaBike_trends.drop("Week", axis=1, inplace=True)


# %%

ax = hondaBike_trends["Search"].plot(title='Honda Bike Search Trend', **plot_params)
_ = ax.set(ylabel="Honda Bike Search Trend")

# %%
# Add google search results in Honda daily sales file

ax = hondaBike_trends.plot(
    y=["Search", "ScaledSales"]
)
_ = ax.set(title="Google Search (Honda) Weekly Trend and Honda Weekly Retail Scaled Sales")

# %%
BikeBrands_trends = pd.read_csv("D:/Python/Input/GoogleTrends/TempComparison/Bike brand search 2017-9.14.22.csv")
BikeBrands_trends.set_index(
    pd.PeriodIndex(BikeBrands_trends.Week, freq="W"),
    inplace=True,
)
BikeBrands_trends.drop("Week", axis=1, inplace=True)

ax = BikeBrands_trends.plot(
    y=["Honda", "Yamaha", "Suzuki", "Bajaj", "TVS"]
)

_ = ax.set(title="Google Search Trends of Motorcycle Brands")

# %%
CBR15_trends = pd.read_csv("D:/Python/Input/GoogleTrends/TempComparison/cbr r15 17-14.9.22.csv")
CBR15_trends.set_index(
    pd.PeriodIndex(CBR15_trends.Week, freq="W"),
    inplace=True,
)
CBR15_trends.drop("Week", axis=1, inplace=True)

ax = CBR15_trends.plot(
    y=["CBR", "R15"]
)

_ = ax.set(title="Google Search Trends of CBR and R15")

# %%
cc110_trends = pd.read_csv("D:/Python/Input/GoogleTrends/TempComparison/dream livo discover metro 17-9.14.22.csv")
cc110_trends.set_index(
    pd.PeriodIndex(cc110_trends.Week, freq="W"),
    inplace=True,
)
cc110_trends.drop("Week", axis=1, inplace=True)

ax = cc110_trends.plot(
    y=["HondaDream", "HondaLivo", "BajajDiscover", "TVSMetro"]
)

_ = ax.set(title="Google Search Trend of 110CC Motorcycles")

# %%
highcc_trends = pd.read_csv("D:/Python/Input/GoogleTrends/TempComparison/xblade hornet pulsar gixxer 2017-14.9.22.csv")
highcc_trends.set_index(
    pd.PeriodIndex(highcc_trends.Week, freq="W"),
    inplace=True,
)
highcc_trends.drop("Week", axis=1, inplace=True)

ax = highcc_trends.plot(
    y=["HondaXblade", "HondaHornet", "BajajPulsar", "SuzukiGixxer"]
)

_ = ax.set(title="Google Search Trend of HighCC Motorcycles")

# %%
def crosscorr(datax, datay, lag=0):
    return datax.corr(datay.shift(lag))

# %%
xcov_monthly = [crosscorr(hondaBike_trends.Search, hondaBike_trends.ScaledSales, lag=i) for i in range(30)]

xcov_monthly

# %%
_ = plot_lags(hondaBike_trends.Search, lags=12, nrows=2)
_ = plot_pacf(hondaBike_trends.ScaledSales, lags=12)

# %%
#Correlation on moving average

moving_average = hondaBike_trends.rolling(
    window=12,       # 6-month window
    center=True,      # puts the average at the center of the window
    min_periods=6,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

ax = hondaBike_trends.plot(style=".", color="0.5", legend = False)
moving_average.plot(
    ax=ax, linewidth=3, title="Google Search (Honda) Trend and Honda Retail Scaled Sales - 12 week Moving Average", legend=True,
);

# %%


# %% [markdown]
# ### Time Series as Feature

# %%
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import periodogram
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf

simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)
%config InlineBackend.figure_format = 'retina'


def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig

# %%
_ = plot_lags(monthlySales.HondaSales, lags=12, nrows=2)
_ = plot_pacf(monthlySales.HondaSales, lags=12)

# %%
_ = plot_lags(hondaBike_trends.Search, hondaBike_trends.ScaledSales, lags=36, nrows=6)
_ = plot_pacf(hondaBike_trends.Search, hondaBike_trends.ScaledSales, lags=12)

# %%
def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)

X = make_lags(monthlySales.HondaSales, lags=1)
X = X.fillna(0.0)

# %%
# Create target series and data splits
y = monthlySales.HondaSales.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=60, shuffle=False)

# Fit and predict
model = LinearRegression()  # `fit_intercept=True` since we didn't use DeterministicProcess
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_train), index=y_train.index)
y_fore = pd.Series(model.predict(X_test), index=y_test.index)

# %%
ax = y_train.plot(**plot_params)
ax = y_test.plot(**plot_params)
ax = y_pred.plot(ax=ax)
_ = y_fore.plot(ax=ax, color='C3')

# %%
ax = y_test.plot(**plot_params)
_ = y_fore.plot(ax=ax, color='C3')

# %%
#Read google trend data

data_dir = Path("D:/Python/Input/GoogleTrends/")
hondaBike_trends = pd.read_csv(data_dir / "HondaBikeSearchTrend.csv")
hondaBike_trends.set_index(
    pd.PeriodIndex(hondaBike_trends.Month, freq="M"),
    inplace=True,
)
hondaBike_trends.drop("Month", axis=1, inplace=True)

ax = hondaBike_trends.hondaBikeSearch.plot(title='Honda Bike Search Trend', **plot_params)
_ = ax.set(ylabel="Honda Bike Search Trend")

# %%
# Add google search results in Honda daily sales file

ax = hondaBike_trends.plot(
    y=["hondaBikeSearch", "NormalizedHondaSales"]
)

# %%
ax = Honda17Q1_17Q2Daily["Honda: (Bangladesh)"].plot(title='Honda Bike Search Trend', **plot_params)
_ = ax.set(ylabel="Honda Bike Search Trend")

# %% [markdown]
# ## Trend

# %% [markdown]
# To update

# %%
search_terms = ["FluContagious", "FluCough", "FluFever", "InfluenzaA", "TreatFlu", "IHaveTheFlu", "OverTheCounterFlu", "HowLongFlu"]

# Create three lags for each search term
X0 = make_lags(flu_trends[search_terms], lags=3)

# Create four lags for the target, as before
X1 = make_lags(flu_trends['FluVisits'], lags=4)

# Combine to create the training data
X = pd.concat([X0, X1], axis=1).fillna(0.0)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=60, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_train), index=y_train.index)
y_fore = pd.Series(model.predict(X_test), index=y_test.index)

ax = y_test.plot(**plot_params)
_ = y_fore.plot(ax=ax, color='C3')

# %% [markdown]
# ### Discarded code backup

# %%
# scaleVal2 = (Honda17Q3_17Q4DailyScaled.iloc[-1,0]) / (Honda18Q1_18Q2Daily.iloc[0,0])
# Honda18Q1_18Q2DailyScaled = Honda18Q1_18Q2Daily.select_dtypes(exclude=['object', 'datetime']) * scaleVal2


# %%
# #automating trial of scaling all search trend

# searchTrendfiles = [Honda17Q1_17Q2Daily, Honda17Q3_17Q4Daily,
#                     Honda18Q1_18Q2Daily,
#                     Honda18Q3_18Q4Daily,
#                     Honda19Q1_19Q2Daily,
#                     Honda19Q3_19Q4Daily,
#                     Honda20Q1_20Q2Daily,
#                     Honda20Q3_20Q4Daily,
#                     Honda21Q1_21Q2Daily,
#                     Honda21Q3_21Q4Daily,
#                     Honda22Q1_22Q2Daily,
#                     Honda22Q3_current]

# scaledTrend = []

# for i in searchTrendfiles:
#     scaleVal = (i.iloc[-1,0]) / ((i+1).iloc[0,0])
#     scaledCol = pd.DataFrame(scaleVal * (i+1).iloc[:,0])
#     scaledCol = scaledCol.iloc[:,0].round(1)
#     scaledTrend.append(scaledCol)


# %%
# scaleVal1 = (Honda17Q1_17Q2Daily.iloc[-1,0]) / (Honda17Q3_17Q4Daily.iloc[0,0])
# Honda17Q3_17Q4DailyScaled = scaleVal1 * Honda17Q3_17Q4Daily.iloc[:,0]
# Honda17Q3_17Q4DailyScaled = pd.DataFrame(Honda17Q3_17Q4DailyScaled.round(1))

# scaleVal2 = (Honda17Q3_17Q4DailyScaled.iloc[-1,0]) / (Honda18Q1_18Q2Daily.iloc[0,0])
# Honda18Q1_18Q2DailyScaled = pd.DataFrame(scaleVal2 * Honda18Q1_18Q2Daily.iloc[:,0])
# Honda18Q1_18Q2DailyScaled = Honda18Q1_18Q2DailyScaled.iloc[:,0].round(1)

# scaleVal3 = (Honda18Q1_18Q2DailyScaled.iloc[-1,0]) / (Honda18Q3_18Q4Daily.iloc[0,0])
# Honda18Q3_18Q4DailyScaled = pd.DataFrame(scaleVal3 * Honda18Q3_18Q4Daily.iloc[:,0])
# Honda18Q3_18Q4DailyScaled = Honda18Q3_18Q4DailyScaled.iloc[:,0].round(1)

# scaleVal4 = (Honda18Q3_18Q4DailyScaled.iloc[-1,0]) / (Honda19Q1_19Q2Daily.iloc[0,0])
# Honda19Q1_19Q2DailyScaled = pd.DataFrame(scaleVal4 * Honda19Q1_19Q2Daily.iloc[:,0])
# Honda19Q1_19Q2DailyScaled = Honda19Q1_19Q2DailyScaled.iloc[:,0].round(1)

# scaleVal5 = (Honda19Q1_19Q2DailyScaled.iloc[-1,0]) / (Honda19Q3_19Q4Daily.iloc[0,0])
# Honda19Q3_19Q4DailyScaled = pd.DataFrame(scaleVal5 * Honda19Q3_19Q4Daily.iloc[:,0])
# Honda19Q3_19Q4DailyScaled = Honda19Q3_19Q4DailyScaled.iloc[:,0].round(1)

# scaleVal6 = (Honda19Q3_19Q4DailyScaled.iloc[-1,0]) / (Honda20Q1_20Q2Daily.iloc[0,0])
# Honda20Q1_20Q2DailyScaled = pd.DataFrame(scaleVal6 * Honda20Q1_20Q2Daily.iloc[:,0])
# Honda20Q1_20Q2DailyScaled = Honda20Q1_20Q2DailyScaled.iloc[:,0].round(1)

# scaleVal7 = (Honda20Q1_20Q2DailyScaled.iloc[-1,0]) / (Honda20Q3_20Q4Daily.iloc[0,0])
# Honda20Q3_20Q4DailyScaled = pd.DataFrame(scaleVal7 * Honda20Q3_20Q4Daily.iloc[:,0])
# Honda20Q3_20Q4DailyScaled = Honda20Q3_20Q4DailyScaled.iloc[:,0].round(1)

# scaleVal8 = (Honda20Q3_20Q4DailyScaled.iloc[-1,0]) / (Honda21Q1_21Q2Daily.iloc[0,0])
# Honda21Q1_21Q2DailyScaled = pd.DataFrame(scaleVal8 * Honda21Q1_21Q2Daily.iloc[:,0])
# Honda21Q1_21Q2DailyScaled = Honda21Q1_21Q2DailyScaled.iloc[:,0].round(1)

# scaleVal9 = (Honda21Q1_21Q2DailyScaled.iloc[-1,0]) / (Honda21Q3_21Q4Daily.iloc[0,0])
# Honda21Q3_21Q4DailyScaled = pd.DataFrame(scaleVal9 * Honda21Q3_21Q4Daily.iloc[:,0])
# Honda21Q3_21Q4DailyScaled = Honda21Q3_21Q4DailyScaled.iloc[:,0].round(1)

# scaleVal10 = (Honda21Q3_21Q4DailyScaled.iloc[-1,0]) / (Honda22Q1_22Q2Daily.iloc[0,0])
# Honda22Q1_22Q2DailyScaled = pd.DataFrame(scaleVal10 * Honda22Q1_22Q2Daily.iloc[:,0])
# Honda22Q1_22Q2DailyScaled = Honda22Q1_22Q2DailyScaled.iloc[:,0].round(1)

# scaleVal11 = (Honda22Q1_22Q2DailyScaled.iloc[-1,0]) / (Honda22Q3_current.iloc[0,0])
# Honda22Q3_currentScaled = pd.DataFrame(scaleVal11 * Honda22Q3_current.iloc[:,0])
# Honda22Q3_currentScaled = Honda22Q3_currentScaled.iloc[:,0].round(1)



# %%
# scaleVal2 = (Honda17Q3_17Q4DailyScaled.iloc[-1,0]) / (Honda18Q1_18Q2Daily.iloc[0,0])
# Honda18Q1_18Q2DailyScaled = pd.DataFrame(scaleVal2 * Honda18Q1_18Q2Daily.iloc[:,0])


# scaleVal3 = (Honda18Q1_18Q2DailyScaled.iloc[-1,0]) / (Honda18Q3_18Q4Daily.iloc[0,0])
# Honda18Q3_18Q4DailyScaled = pd.DataFrame(scaleVal3 * Honda18Q3_18Q4Daily.iloc[:,0])


# scaleVal4 = (Honda18Q3_18Q4DailyScaled.iloc[-1,0]) / (Honda19Q1_19Q2Daily.iloc[0,0])
# Honda19Q1_19Q2DailyScaled = pd.DataFrame(scaleVal4 * Honda19Q1_19Q2Daily.iloc[:,0])


# scaleVal5 = (Honda19Q1_19Q2DailyScaled.iloc[-1,0]) / (Honda19Q3_19Q4Daily.iloc[0,0])
# Honda19Q3_19Q4DailyScaled = Honda19Q3_19Q4Daily.iloc[:,0] * scaleVal5


# scaleVal6 = (Honda19Q3_19Q4DailyScaled.iloc[-1,0]) / (Honda20Q1_20Q2Daily.iloc[0,0])
# Honda20Q1_20Q2DailyScaled = Honda20Q1_20Q2Daily.iloc[:,0] * scaleVal6


# scaleVal7 = (Honda20Q1_20Q2DailyScaled.iloc[-1,0]) / (Honda20Q3_20Q4Daily.iloc[0,0])
# Honda20Q3_20Q4DailyScaled = Honda20Q3_20Q4Daily.iloc[:,0] * scaleVal7


# scaleVal8 = (Honda20Q3_20Q4DailyScaled.iloc[-1,0]) / (Honda21Q1_21Q2Daily.iloc[0,0])
# Honda21Q1_21Q2DailyScaled = Honda21Q1_21Q2Daily.iloc[:,0] * scaleVal8


# scaleVal9 = (Honda21Q1_21Q2DailyScaled.iloc[-1,0]) / (Honda21Q3_21Q4Daily.iloc[0,0])
# Honda21Q3_21Q4DailyScaled = Honda21Q3_21Q4Daily.iloc[:,0] * scaleVal9


# scaleVal10 = (Honda21Q3_21Q4DailyScaled.iloc[-1,0]) / (Honda22Q1_22Q2Daily.iloc[0,0])
# Honda22Q1_22Q2DailyScaled = Honda22Q1_22Q2Daily.iloc[:,0] * scaleVal10


# scaleVal11 = (Honda22Q1_22Q2DailyScaled.iloc[-1,0]) / (Honda22Q3_current.iloc[0,0])
# Honda22Q3_currentScaled = Honda22Q3_current.iloc[:,0] * scaleVal11

# %%
# scaledHonda17Q1_CurrentDaily["Honda: (Bangladesh)"] = int(scaledHonda17Q1_CurrentDaily["Honda: (Bangladesh)"])


