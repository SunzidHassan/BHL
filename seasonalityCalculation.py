# %% [markdown]
# ## Setup

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
# ### Plot defaults

# %%
# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(10, 5))
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
# ## Seasonal Function

# %% [markdown]
# ### Function

# %%
def seasonal_weight(data_dir, sheet_name, dateCol, timeFrame, seasonalVar):
    """
    Return seasonal weights of the seasonal var within the timeframe
    
    >>> data_dir = Path("D:/Python/Input/HondaData/IndependentVarsPy.xlsx")
    >>> sheet_name = "GregorianHirjiCalendarSales"
    >>> dateCol = "GregorianDate"
    >>> timeFrame = "HijriYear"
    >>> seasonalVar = "HijriMonth"
    >>> hijrimonthPerYearSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, timeFrame, seasonalVar).mean()

    Sales  1             0.834862
       2             0.841673
       3             0.795200
       4             0.796001
       5             0.957686
       6             1.056326
       7             1.036962
       8             0.971593
       9             1.405286
       10            1.005048
       11            0.897577
       12            1.401788
    dtype: float64
    """

    # dataload
    dailySales = pd.read_excel(data_dir,
                            sheet_name = sheet_name,
                            parse_dates=[dateCol])

    #dailyHijriSales = dailyHijriSales.rename(columns={'0': 'Amount'})
    #dailyHijriSales = dailyHijriSales.loc[dailyHijriSales['GregorianDate'] > '2017-09-01']

    dailySales = dailySales.set_index(dateCol).to_period(freq='D')

    #Discard bad years
    salesData = dailySales.loc[(dailySales.HijriYear == 1339) |
                                (dailySales.HijriYear == 1440) |
                                (dailySales.HijriYear == 1442) |
                                (dailySales.HijriYear == 1443)]
    
    timeFrame = salesData[timeFrame]
    seasonalVar = salesData[seasonalVar]

    #Pivot Data
    Pivot = salesData.pivot_table(index = timeFrame, columns= seasonalVar, values=["Sales"], aggfunc = 'sum')

    
    # Small seasonality within defined time frame
    seasonalChange = Pivot.copy()
    seasonalChange.Sales = 0
    seasonalChange.Sales.iloc[:,0] = 1

    for longRange in Pivot.index:
        for shortRange in range(1, len(Pivot.columns)):
            seasonalChange.loc[longRange, ('Sales', shortRange+1)] = 1 + ((Pivot.loc[longRange, ('Sales', shortRange+1)] - Pivot.loc[longRange, ('Sales', shortRange)]) / Pivot.loc[longRange, ('Sales', shortRange)])
    seasonalChange

    # Cumulative change
    cumulativeChange = seasonalChange.copy()
    cumulativeChange.Sales[2:len(seasonalChange.columns)] = 0

    for longRange in seasonalChange.index:
        for shortRange in range(1, len(seasonalChange.columns)):
            cumulativeChange.loc[longRange, ('Sales', shortRange+1)] = cumulativeChange.loc[longRange, ('Sales', shortRange)] * cumulativeChange.loc[longRange, ('Sales', shortRange+1)]

    cumulativeChangeSum = cumulativeChange.sum(axis=1)

    #Total sales of the first month
    firstCycleTTLSales = Pivot.iloc[0,:].sum()

    #Constant total sales
    constantTTLShortRangeSales = cumulativeChange.copy()
    constantTTLShortRangeSales.Sales[2:len(seasonalChange.columns)] = 0

    for longRange in seasonalChange.index:
        for shortRange in range(1, len(seasonalChange.columns)+1):
            constantTTLShortRangeSales.loc[longRange, ('Sales', shortRange)] = (firstCycleTTLSales / cumulativeChangeSum[longRange]) * cumulativeChange.loc[longRange, ('Sales', shortRange)]

    # Uniform scale
    firstLongrangeTTLSales = Pivot.iloc[0,:].sum()

    shortrangeSeasonalWeight = constantTTLShortRangeSales.copy()
    shortrangeSeasonalWeight.Sales[2:len(seasonalChange.columns)] = 0

    for longRange in seasonalChange.index:
        for shortRange in range(1, len(seasonalChange.columns)+1):
            shortrangeSeasonalWeight.loc[longRange, ('Sales', shortRange)] = (len(seasonalChange.columns) / firstLongrangeTTLSales) * constantTTLShortRangeSales.loc[longRange, ('Sales', shortRange)]

    # for monthday scale, if possible, address 29/30 day issue, considering that columns will be 30
    # if seasonalVar == salesData["HijriDayPerMonth"]:
    #     pass

    return shortrangeSeasonalWeight


# %% [markdown]
# ### Function output

# %% [markdown]
# #### Hijri month seasonality in Hijri year

# %%
data_dir = Path("D:/Python/Input/HondaData/IndependentVarsPy.xlsx")
sheet_name = "GregorianHirjiCalendarSales"
dateCol = "GregorianDate"
timeFrame = "HijriYear"
seasonalVar = "HijriMonth"

#Seasonal function
hijrimonthPerYearSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, timeFrame, seasonalVar).mean()

print(hijrimonthPerYearSeasonalWeight)

#Plot weights
ax = hijrimonthPerYearSeasonalWeight.Sales.plot(linewidth = 3, color="0.5", title="Seasonal weight")

# %% [markdown]
# #### Weekday seasonality in Hijri Month

# %%
data_dir = Path("D:/Python/Input/HondaData/IndependentVarsPy.xlsx")
sheet_name = "GregorianHirjiCalendarSales"
dateCol = "GregorianDate"
timeFrame = "HijriMonth"
seasonalVar = "WeekdayNum"

#Seasonal function
WeekdayperMonthSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, timeFrame, seasonalVar).mean()

print(WeekdayperMonthSeasonalWeight)

#Plot weights
ax = WeekdayperMonthSeasonalWeight.Sales.plot(linewidth = 3, color="0.5", title="Seasonal weight")

# %% [markdown]
# #### Monthday seasonality in Hijri Months

# %%
data_dir = Path("D:/Python/Input/HondaData/IndependentVarsPy.xlsx")
sheet_name = "GregorianHirjiCalendarSales"
dateCol = "GregorianDate"
timeFrame = "HijriMonth"
seasonalVar = "HijriDayPerMonth"

#Seasonal function
monthdayPerMonthSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, timeFrame, seasonalVar)

#Plot weights
ax = monthdayPerMonthSeasonalWeight.Sales.iloc[8, :].plot(linewidth = 1, color=".4", title="Seasonal weight")
ax = monthdayPerMonthSeasonalWeight.Sales.iloc[11, :].plot(linewidth = 1, color=".8", title="Seasonal weight")


# %% [markdown]
# ## Hijri seasonality to Gregorian forecast function

# %% [markdown]
# ### Function

# %%
def seasonal_weight(data_dir, sheet_name, dateCol, timeFrame, seasonalVar):
    """
    Return seasonal weights of the seasonal var within the timeframe
    
    >>> data_dir = Path("D:/Python/Input/HondaData/IndependentVarsPy.xlsx")
    >>> sheet_name = "GregorianHirjiCalendarSales"
    >>> dateCol = "GregorianDate"
    >>> timeFrame = "HijriYear"
    >>> seasonalVar = "HijriMonth"
    >>> hijrimonthPerYearSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, timeFrame, seasonalVar).mean()

    Sales  1             0.834862
       2             0.841673
       3             0.795200
       4             0.796001
       5             0.957686
       6             1.056326
       7             1.036962
       8             0.971593
       9             1.405286
       10            1.005048
       11            0.897577
       12            1.401788
    dtype: float64
    """

    # dataload
    dailySales = pd.read_excel(data_dir,
                            sheet_name = sheet_name,
                            parse_dates=[dateCol])

    #dailyHijriSales = dailyHijriSales.rename(columns={'0': 'Amount'})
    #dailyHijriSales = dailyHijriSales.loc[dailyHijriSales['GregorianDate'] > '2017-09-01']

    dailySales = dailySales.set_index(dateCol).to_period(freq='D')

    #Discard bad years
    salesData = dailySales.loc[(dailySales.HijriYear == 1339) |
                                (dailySales.HijriYear == 1440) |
                                (dailySales.HijriYear == 1442) |
                                (dailySales.HijriYear == 1443)]
    
    timeFrame = salesData[timeFrame]
    seasonalVar = salesData[seasonalVar]

    #Pivot Data
    Pivot = salesData.pivot_table(index = timeFrame, columns= seasonalVar, values=["Sales"], aggfunc = 'sum')

    
    # Small seasonality within defined time frame
    seasonalChange = Pivot.copy()
    seasonalChange.Sales = 0
    seasonalChange.Sales.iloc[:,0] = 1

    for longRange in Pivot.index:
        for shortRange in range(1, len(Pivot.columns)):
            seasonalChange.loc[longRange, ('Sales', shortRange+1)] = 1 + ((Pivot.loc[longRange, ('Sales', shortRange+1)] - Pivot.loc[longRange, ('Sales', shortRange)]) / Pivot.loc[longRange, ('Sales', shortRange)])
    
    # Cumulative change
    cumulativeChange = seasonalChange.copy()
    cumulativeChange.Sales[2:len(seasonalChange.columns)] = 0

    for longRange in seasonalChange.index:
        for shortRange in range(1, len(seasonalChange.columns)):
            cumulativeChange.loc[longRange, ('Sales', shortRange+1)] = cumulativeChange.loc[longRange, ('Sales', shortRange)] * cumulativeChange.loc[longRange, ('Sales', shortRange+1)]

    cumulativeChangeSum = cumulativeChange.sum(axis=1)

    #Total sales of the first month
    firstCycleTTLSales = Pivot.iloc[0,:].sum()

    #Constant total sales
    constantTTLShortRangeSales = cumulativeChange.copy()
    constantTTLShortRangeSales.Sales[2:len(seasonalChange.columns)] = 0

    for longRange in seasonalChange.index:
        for shortRange in range(1, len(seasonalChange.columns)+1):
            constantTTLShortRangeSales.loc[longRange, ('Sales', shortRange)] = (firstCycleTTLSales / cumulativeChangeSum[longRange]) * cumulativeChange.loc[longRange, ('Sales', shortRange)]

    # Uniform scale
    firstLongrangeTTLSales = Pivot.iloc[0,:].sum()

    shortrangeSeasonalWeight = constantTTLShortRangeSales.copy()
    shortrangeSeasonalWeight.Sales[2:len(seasonalChange.columns)] = 0

    for longRange in seasonalChange.index:
        for shortRange in range(1, len(seasonalChange.columns)+1):
            shortrangeSeasonalWeight.loc[longRange, ('Sales', shortRange)] = (len(seasonalChange.columns) / firstLongrangeTTLSales) * constantTTLShortRangeSales.loc[longRange, ('Sales', shortRange)]

    # for monthday scale, if possible, address 29/30 day issue, considering that columns will be 30
    # if seasonalVar == salesData["HijriDayPerMonth"]:
    #     pass

    return shortrangeSeasonalWeight

def gregorianForecast(data_dir, sheet_name, dateCol, startDate, endDate, salesTarget):
    

    dailySales = pd.read_excel(data_dir,
                                sheet_name = sheet_name,
                                parse_dates=[dateCol])

    dailySales = dailySales#.set_index(dateCol).to_period(freq='D')

    calenderMergeSource = dailySales.loc[:, (dateCol, 'HijriYear', 'HijriMonth', 'HijriDayPerMonth', 'WeekdayNum', 'Sales')]


    ##Load seasonality from function and convert list that you received from seasonality function into dataframe - so that it can be merged

    # month per year weight
    hijrimonthPerYearSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, "HijriYear", "HijriMonth").mean()
    hijrimonthPerYearSeasonalWeightDF = pd.DataFrame(hijrimonthPerYearSeasonalWeight, columns=['monthPerYearWeight'])

    #weekday weight
    WeekdayperMonthSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, "HijriMonth", "WeekdayNum").mean()
    WeekdayperMonthSeasonalWeightDF = pd.DataFrame(WeekdayperMonthSeasonalWeight, columns=['weekdayPerMonthWeight'])

    #monthday weight
    old = seasonal_weight(data_dir, sheet_name, dateCol, "HijriMonth", "HijriDayPerMonth")
    new = pd.DataFrame(index=range(361), columns=['HijriMonth', 'HijriDayPerMonth', 'monthdayPerMonthWeight'])

    for i in range(1, 13):
        for j in range(1, 31):
            row = ((i-1)*30)+j
            new['HijriMonth'].iloc[row] = i
            new['HijriDayPerMonth'].iloc[row] = j
            new['monthdayPerMonthWeight'].iloc[row] = old.loc[i, ('Sales', j)]

    monthdayPerMonthSeasonalWeight = new.iloc[1:,:]

    # join 3 weights with the gregorian calendar
    monthWeightJoin = pd.merge(calenderMergeSource.reset_index(), 
                        hijrimonthPerYearSeasonalWeightDF, 
                        on ='HijriMonth', 
                        how ='left')

    weekdayWeightJoin = pd.merge(monthWeightJoin, 
                        WeekdayperMonthSeasonalWeightDF, 
                        on ='WeekdayNum', 
                        how ='left')

    monthdayWeightJoin = pd.merge(weekdayWeightJoin, 
                        monthdayPerMonthSeasonalWeight, 
                        on =['HijriMonth', 'HijriDayPerMonth'], 
                        how ='left')

    combinedWeight = monthdayWeightJoin.copy()
    #combinedWeight.to_csv(outputPath/ "combinedWeight10.17.22.csv")

    monthWeight = combinedWeight.loc[(combinedWeight.GregorianDate.dt.date > startDate) & (combinedWeight.GregorianDate.dt.date < endDate)][['GregorianDate', 'HijriYear', 'HijriMonth', 'monthPerYearWeight']]

    monthWeightPivot = monthWeight.pivot_table(index = (monthWeight.HijriYear, monthWeight.HijriMonth), values=["monthPerYearWeight"], aggfunc = ('mean', 'count'))

    # new month weights for given total weight scale. If month = 12 and day = 30, then new weight = old weight
    for years in monthWeight.HijriYear.unique():
            for months in monthWeightPivot.loc[years].index:
                monthWeightPivot.loc[(years, months), ('monthPerYearWeight', 'newMean')] = monthWeightPivot.loc[(years, months), ('monthPerYearWeight', 'mean')] * (monthWeightPivot.loc[(years, months), ('monthPerYearWeight', 'count')] / 30)

    totalNewMonthlyWeight = monthWeightPivot.loc[:, ('monthPerYearWeight', 'newMean')].sum()

    for years in monthWeight.HijriYear.unique():
            for months in monthWeightPivot.loc[years].index:
                monthWeightPivot.loc[(years, months), ('monthPerYearWeight', 'monthSales')] = round(salesTarget * (monthWeightPivot.loc[(years, months), ('monthPerYearWeight', 'newMean')] / totalNewMonthlyWeight),0)


    # new monthday weight
    monthdayWeight = combinedWeight.loc[(combinedWeight.GregorianDate.dt.date > startDate) & (combinedWeight.GregorianDate.dt.date < endDate)][['HijriYear', 'HijriMonth', 'HijriDayPerMonth', 'monthdayPerMonthWeight', 'weekdayPerMonthWeight']]

    #monthdayWeightPivot = monthdayWeight.pivot_table(index = (monthdayWeight.HijriYear, monthdayWeight.HijriMonth, monthdayWeight.HijriDayPerMonth), values=["monthdayPerMonthWeight", "weekdayPerMonthWeight"], aggfunc = ('mean'))
    monthdayWeightPivot1 = monthdayWeight.pivot_table(index = (monthdayWeight.HijriYear, monthdayWeight.HijriMonth, monthdayWeight.HijriDayPerMonth), values=['monthdayPerMonthWeight'], aggfunc = 'mean')
    monthdayWeightPivot2 = monthdayWeight.pivot_table(index = (monthdayWeight.HijriYear, monthdayWeight.HijriMonth, monthdayWeight.HijriDayPerMonth), values=['weekdayPerMonthWeight'], aggfunc = 'mean')
    monthdayWeightPivot = pd.merge(monthdayWeightPivot1, 
                        monthdayWeightPivot2, 
                        on = ('HijriYear', 'HijriMonth', 'HijriDayPerMonth'), 
                        how ='left')

    for years in monthWeight.HijriYear.unique():
            for months in monthdayWeightPivot.loc[years].index.get_level_values(0).unique():
                for monthDays in monthdayWeightPivot.loc[years, months].index:
                    monthdayWeightPivot.loc[(years, months, monthDays), 'newMonthdayWeight'] = (monthdayWeightPivot.loc[(years, months, monthDays), 'monthdayPerMonthWeight']) * (monthdayWeightPivot.loc[(years, months, monthDays), 'weekdayPerMonthWeight'])

    for years in monthWeight.HijriYear.unique():
            for months in monthdayWeightPivot.loc[years].index.get_level_values(0).unique():
                monthdayWeightPivot.loc[(years, months), 'monthWeightTotal'] = monthdayWeightPivot.loc[(years, months), 'newMonthdayWeight'].sum()

    for years in monthWeight.HijriYear.unique():
            for months in monthdayWeightPivot.loc[years].index.get_level_values(0).unique():
                for monthDays in monthdayWeightPivot.loc[years, months].index:
                    monthdayWeightPivot.loc[(years, months, monthDays), 'sales'] = round(((monthWeightPivot.loc[(years, months), ('monthPerYearWeight', 'monthSales')]) / (monthdayWeightPivot.loc[(years, months, monthDays), 'monthWeightTotal'])) * (monthdayWeightPivot.loc[(years, months, monthDays), 'newMonthdayWeight']),0)

    #Gregorian sum
    targetCalender = calenderMergeSource.reset_index()
    targetCalender = targetCalender.loc[(targetCalender.GregorianDate.dt.date > startDate) & (targetCalender.GregorianDate.dt.date < endDate)]

    targetGregorianCalendarSum = pd.merge(targetCalender,
                        monthdayWeightPivot.loc[:, 'sales'].reset_index(), 
                        on = ['HijriYear', 'HijriMonth', 'HijriDayPerMonth'], 
                        how ='left')

    targetGregorianCalendarSum['GregorianYear'] = targetGregorianCalendarSum.GregorianDate.dt.year
    targetGregorianCalendarSum['GregorianMonth'] = targetGregorianCalendarSum.GregorianDate.dt.month


    targetGregorianCalendarMonthlySum = targetGregorianCalendarSum.pivot_table(index=('GregorianYear', 'GregorianMonth'), values=["sales"], aggfunc='sum')
    targetGregorianCalendarMonthlySum = targetGregorianCalendarMonthlySum.reset_index()

    return targetGregorianCalendarSum, targetGregorianCalendarMonthlySum

# %% [markdown]
# ### Todo: Standard deviation, Range prediction and Fan chart

# %%


# %% [markdown]
# ### Output

# %% [markdown]
# #### Month Forecast

# %%
#Load seasonal function first
data_dir = Path("D:/Python/Input/HondaData/IndependentVarsPy.xlsx")
sheet_name = "GregorianHirjiCalendarSales"
dateCol = "GregorianDate"
startDate19 = dt.date(2019,4,1)
endDate19 = dt.date(2020,3,31)
startDate20 = dt.date(2020,4,1)
endDate20 = dt.date(2021,3,31)
salesTarget = 90000

#Monthly forecast of two year
monthForecast19 = gregorianForecast(data_dir, sheet_name, dateCol, startDate19, endDate19, salesTarget)[1]
monthForecast20 = gregorianForecast(data_dir, sheet_name, dateCol, startDate20, endDate20, salesTarget)[1]

print(monthForecast19)
print(monthForecast20)


ax1 = monthForecast19.sales.plot(linewidth = 3, color='c', title="Month Forecast", label = '2019', legend=True)
ax2 = monthForecast20.sales.plot(linewidth = 3, color='b', label = '2020', legend=True)

# %% [markdown]
# #### Day forecast

# %%
#Load seasonal function first
data_dir = Path("D:/Python/Input/HondaData/IndependentVarsPy.xlsx")
sheet_name = "GregorianHirjiCalendarSales"
dateCol = "GregorianDate"
startDate19 = dt.date(2019,4,1)
endDate19 = dt.date(2020,3,31)
startDate20 = dt.date(2020,4,1)
endDate20 = dt.date(2021,3,31)
salesTarget = 90000

#Yearly forecast of two year
dayForecast19 = gregorianForecast(data_dir, sheet_name, dateCol, startDate19, endDate19, salesTarget)[0]
dayForecast20 = gregorianForecast(data_dir, sheet_name, dateCol, startDate20, endDate20, salesTarget)[0]

print(dayForecast19)
print(dayForecast20)

ax1 = dayForecast19.sales.plot(linewidth = 3, color='c', title="Month Forecast", label = '2019', legend=True)
ax2 = dayForecast20.sales.plot(linewidth = 3, color='b', label = '2020', legend=True)
