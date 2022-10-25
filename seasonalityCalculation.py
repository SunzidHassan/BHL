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
def seasonal_weight(data_dir, sheet_name, dateCol, salesVar, timeFrame, seasonalVar):
    """
    Return seasonal weights of the seasonal var within the timeframe
    
    >>> data_dir = Path("D:/Python/Input/HondaData/IndependentVarsPy.xlsx")
    >>> sheet_name = "GregorianHirjiCalendarSales"
    >>> dateCol = "GregorianDate"
    >>> salesVar = "BHLRetailSales"
    >>> timeFrame = "HijriYear"
    >>> seasonalVar = "HijriMonth"
    >>> hijrimonthPerYearSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, salesVar, timeFrame, seasonalVar).mean()

    Sales  1         0.834862
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

    dailySales = dailySales.set_index(dateCol).to_period(freq='D')

    #Discard bad years
    salesData = dailySales.loc[(dailySales.HijriYear == 1339) |
                                (dailySales.HijriYear == 1440) |
                                (dailySales.HijriYear == 1442) |
                                (dailySales.HijriYear == 1443)]
    
    #convert function arguments
    timeFrame = salesData[timeFrame]
    seasonalVar = salesData[seasonalVar]
    Sales = salesData[salesVar]

    salesData.rename(columns = {salesVar:'Sales'}, inplace = True)

    #Pivot Data
    Pivot = salesData.pivot_table(index = timeFrame, columns= seasonalVar, values=['Sales'], aggfunc = 'sum')

    
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

    #Same scale
    shortrangeSeasonalWeight = cumulativeChange.copy()
    shortrangeSeasonalWeight.Sales[2:len(seasonalChange.columns)] = 0

    for longRange in seasonalChange.index:
        for shortRange in range(1, len(seasonalChange.columns)+1):
            shortrangeSeasonalWeight.loc[longRange, ('Sales', shortRange)] = (len(seasonalChange.columns) / cumulativeChangeSum[longRange]) * cumulativeChange.loc[longRange, ('Sales', shortRange)]

    return shortrangeSeasonalWeight


# %% [markdown]
# ### Function output

# %%
data_dir = Path("D:/Python/Input/HondaData/IndependentVarsPy.xlsx")
sheet_name = "GregorianHirjiCalendarSales"
dateCol = "GregorianDate"
salesVar = "BHLRetailSales"

# %% [markdown]
# #### Hijri month seasonality in Hijri year

# %%
timeFrame = "HijriYear"
seasonalVar = "HijriMonth"

#Seasonal function
hijrimonthPerYearSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, salesVar, timeFrame, seasonalVar).mean()

print(hijrimonthPerYearSeasonalWeight)

#Plot weights
ax = hijrimonthPerYearSeasonalWeight.Sales.plot(linewidth = 3, color="0.5", title="Hijri Month Seasonality")

# %% [markdown]
# #### Weekday seasonality in Hijri Month

# %%
timeFrame = "HijriMonth"
seasonalVar = "WeekdayNum"

#Seasonal function
WeekdayperMonthSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, salesVar, timeFrame, seasonalVar).mean()

print(WeekdayperMonthSeasonalWeight)

#Plot weights
ax = WeekdayperMonthSeasonalWeight.Sales.plot(linewidth = 3, color="0.5", title="Weekday seasonality")

plt.xticks([1, 2, 3, 4, 5, 6, 7], ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])

plt.xlabel('Days in Week (Sunday to Saturday)')
plt.ylabel('Daily weight')


# %%


# %% [markdown]
# #### Monthday seasonality in Hijri Months

# %%
timeFrame = "HijriMonth"
seasonalVar = "HijriDayPerMonth"

#Seasonal function
monthdayPerMonthSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, salesVar, timeFrame, seasonalVar)

#Plot weights
ax = monthdayPerMonthSeasonalWeight.Sales.iloc[8, :].plot(linewidth = 1, color='g', label = 'Ramadan Month (Eid-ul-Fitr Sales)', legend=True, title="Hijri Month-day Seasonality")
# = monthdayPerMonthSeasonalWeight.Sales.iloc[11, :].plot(linewidth = 1, color='b', label = 'Dhu al-Hijja Month (Eid-ul-Adha Sales)', legend=True)

plt.xlabel('Days in Month')
plt.ylabel('Daily weight')


# %%
pd.DataFrame(monthdayPerMonthSeasonalWeight.Sales.iloc[11, :]).to_csv("D:/R/SeihanAnalysis/Output/Seasonality/tempMonth5.csv")

# %% [markdown]
# ## Hijri seasonality to Gregorian forecast function

# %% [markdown]
# ### Function

# %%
def gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate, endDate, salesTarget):
    

    dailySales = pd.read_excel(data_dir,
                                sheet_name = sheet_name,
                                parse_dates=[dateCol])

    #dailySales = dailySales#.set_index(dateCol).to_period(freq='D')

    calenderMergeSource = dailySales.loc[:, (dateCol, 'HijriYear', 'HijriMonth', 'HijriDayPerMonth', 'WeekdayNum', salesVar)]
    calenderMergeSource.rename(columns = {salesVar:'Sales'}, inplace = True)

    ##Load seasonality from function and convert list that you received from seasonality function into dataframe - so that it can be merged

    # month per year weight
    hijrimonthPerYearSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, salesVar, "HijriYear", "HijriMonth").mean()
    hijrimonthPerYearSeasonalWeightDF = pd.DataFrame(hijrimonthPerYearSeasonalWeight, columns=['monthPerYearWeight'])

    #weekday weight
    WeekdayperMonthSeasonalWeight = seasonal_weight(data_dir, sheet_name, dateCol, salesVar, "HijriMonth", "WeekdayNum").mean()
    WeekdayperMonthSeasonalWeightDF = pd.DataFrame(WeekdayperMonthSeasonalWeight, columns=['weekdayPerMonthWeight'])

    #monthday weight
    old = seasonal_weight(data_dir, sheet_name, dateCol, salesVar, "HijriMonth", "HijriDayPerMonth")
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

    # new monthday weight
    monthdayWeight = combinedWeight.loc[(combinedWeight.GregorianDate.dt.date > startDate) & (combinedWeight.GregorianDate.dt.date < endDate)][['HijriYear', 'HijriMonth', 'HijriDayPerMonth', 'monthdayPerMonthWeight', 'weekdayPerMonthWeight']]

    #monthdayWeightPivot = monthdayWeight.pivot_table(index = (monthdayWeight.HijriYear, monthdayWeight.HijriMonth, monthdayWeight.HijriDayPerMonth), values=["monthdayPerMonthWeight", "weekdayPerMonthWeight"], aggfunc = ('mean'))
    monthdayWeightPivot1 = monthdayWeight.pivot_table(index = (monthdayWeight.HijriYear, monthdayWeight.HijriMonth, monthdayWeight.HijriDayPerMonth), values=['monthdayPerMonthWeight'], aggfunc = 'mean')
    monthdayWeightPivot2 = monthdayWeight.pivot_table(index = (monthdayWeight.HijriYear, monthdayWeight.HijriMonth, monthdayWeight.HijriDayPerMonth), values=['weekdayPerMonthWeight'], aggfunc = 'mean')
    monthdayWeightPivot = pd.merge(monthdayWeightPivot1, 
                        monthdayWeightPivot2, 
                        on = ('HijriYear', 'HijriMonth', 'HijriDayPerMonth'), 
                        how ='left')

    # calculate new month day weight by multiplying monthday and weekday weights
    for years in monthWeight.HijriYear.unique():
            for months in monthdayWeightPivot.loc[years].index.get_level_values(0).unique():
                for monthDays in monthdayWeightPivot.loc[years, months].index:
                    monthdayWeightPivot.loc[(years, months, monthDays), 'newMonthdayWeight'] = (monthdayWeightPivot.loc[(years, months, monthDays), 'monthdayPerMonthWeight']) * (monthdayWeightPivot.loc[(years, months, monthDays), 'weekdayPerMonthWeight'])

    #calculate total monthday weight of each month
    for years in monthWeight.HijriYear.unique():
            for months in monthdayWeightPivot.loc[years].index.get_level_values(0).unique():
                monthdayWeightPivot.loc[(years, months), 'monthWeightTotal'] = monthdayWeightPivot.loc[(years, months), 'newMonthdayWeight'].sum()

    #keep total monthday weight of each month for the filtered timeline
    totalFilteredMonthdayWeight = monthdayWeightPivot.reset_index(level=2, drop=True)[['monthWeightTotal']].drop_duplicates(keep='first')

    #calculating total possible sum of monthday weights in a month, and sum of monthday weight for filtered days
    tempCombinedWeight = combinedWeight[['HijriYear', 'HijriMonth', 'HijriDayPerMonth','weekdayPerMonthWeight', 'monthdayPerMonthWeight']]
    tempCombinedWeight['newDailyWeight'] = tempCombinedWeight.weekdayPerMonthWeight * tempCombinedWeight.monthdayPerMonthWeight
    totalMonthdayWeight = tempCombinedWeight.pivot_table(index = ('HijriYear', 'HijriMonth'), values=["newDailyWeight"], aggfunc = ('sum'))

    # new month weights for given total weight scale. If month = 12 and day = 30, then new weight = old weight
    for years in monthWeight.HijriYear.unique():
            for months in monthWeightPivot.loc[years].index:
                monthWeightPivot.loc[(years, months), ('monthPerYearWeight', 'newMean')] = monthWeightPivot.loc[(years, months), ('monthPerYearWeight', 'mean')] * ((totalFilteredMonthdayWeight.loc[(years, months), 'monthWeightTotal']) / (totalMonthdayWeight.loc[(years, months), 'newDailyWeight']))

    #calculate sum of all month weight for the given timeline
    totalNewMonthlyWeight = monthWeightPivot.loc[:, ('monthPerYearWeight', 'newMean')].sum()

    #calculate monthly sales from total sales forecast, total month weight and individual monthly weights
    for years in monthWeight.HijriYear.unique():
            for months in monthWeightPivot.loc[years].index:
                monthWeightPivot.loc[(years, months), ('monthPerYearWeight', 'monthSales')] = round(salesTarget * (monthWeightPivot.loc[(years, months), ('monthPerYearWeight', 'newMean')] / totalNewMonthlyWeight),0)

    # forecasting daily sales from monthly sales forecast and daily weights
    for years in monthWeight.HijriYear.unique():
            for months in monthdayWeightPivot.loc[years].index.get_level_values(0).unique():
                for monthDays in monthdayWeightPivot.loc[years, months].index:
                    monthdayWeightPivot.loc[(years, months, monthDays), 'salesForecast'] = round(((monthWeightPivot.loc[(years, months), ('monthPerYearWeight', 'monthSales')]) / (monthdayWeightPivot.loc[(years, months, monthDays), 'monthWeightTotal'])) * (monthdayWeightPivot.loc[(years, months, monthDays), 'newMonthdayWeight']),0)

    #Gregorian sum
    targetCalender = calenderMergeSource.reset_index()
    targetCalender = targetCalender.loc[(targetCalender.GregorianDate.dt.date > startDate) & (targetCalender.GregorianDate.dt.date < endDate)]

    targetGregorianCalendarSum = pd.merge(targetCalender,
                        monthdayWeightPivot.loc[:, 'salesForecast'].reset_index(), 
                        on = ['HijriYear', 'HijriMonth', 'HijriDayPerMonth'], 
                        how ='left')

    targetGregorianCalendarSum['GregorianYear'] = targetGregorianCalendarSum.GregorianDate.dt.year
    targetGregorianCalendarSum['GregorianMonth'] = targetGregorianCalendarSum.GregorianDate.dt.month


    targetGregorianCalendarMonthlySumSales = targetGregorianCalendarSum.pivot_table(index=('GregorianYear', 'GregorianMonth'), values=["Sales"], aggfunc='sum')
    targetGregorianCalendarMonthlySumForecast = targetGregorianCalendarSum.pivot_table(index=('GregorianYear', 'GregorianMonth'), values=["salesForecast"], aggfunc='sum')
    targetGregorianCalendarMonthlySum = pd.merge(targetGregorianCalendarMonthlySumSales.reset_index(),
                                                targetGregorianCalendarMonthlySumForecast.reset_index(),
                                                on = ('GregorianYear', 'GregorianMonth'), 
                                                how ='left')

    #targetGregorianCalendarMonthlySum = targetGregorianCalendarMonthlySum
    #targetGregorianCalendarMonthlySum['GregorianDate'] = dt.date(targetGregorianCalendarMonthlySum.GregorianYear, targetGregorianCalendarMonthlySum.GregorianMonth, 1)

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
salesVar = "BHLRetailSales"

startDate95ki = dt.date(2018,4,1)
endDate95ki = dt.date(2019,3,31)

startDate96ki = dt.date(2019,4,1)
endDate96ki = dt.date(2020,3,31)

startDate97ki = dt.date(2020,4,1)
endDate97ki = dt.date(2021,3,31)

startDate98ki = dt.date(2021,4,1)
endDate98ki = dt.date(2022,3,31)

startDate99ki = dt.date(2022,4,1)
endDate99ki = dt.date(2023,3,31)

startDate100ki = dt.date(2023,4,1)
endDate100ki = dt.date(2024,3,31)

salesTarget95ki = 47626
salesTarget96ki = 58959
salesTarget98ki = 74903
salesTarget99ki1 = 77200
salesTarget99ki2 = 70020
salesTarget100ki = 99500
testTarget = 1000

# %%
data_dir = Path("D:/Python/Input/HondaData/IndependentVarsPy.xlsx")
sheet_name = "GregorianHirjiCalendarSales"
dateCol = "GregorianDate"
salesVar = "BHLRetailSales"

startDate96ki = dt.date(2019,4,1)
endDate96ki = dt.date(2020,3,31)
salesTarget96ki = 58959

monthForecast96ki = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate96ki, endDate96ki, salesTarget96ki)[1]
dayForecast96ki = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate96ki, endDate96ki, salesTarget96ki)[0]


# %%
#Monthly forecast comparison
monthForecast95ki = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate95ki, endDate95ki, salesTarget95ki)[1]
monthForecast96ki = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate96ki, endDate96ki, salesTarget96ki)[1]
monthForecast98ki = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate98ki, endDate98ki, salesTarget98ki)[1]
monthForecast99ki1 = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate99ki, endDate99ki, salesTarget99ki1)[1]
monthForecast99ki2 = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate99ki, endDate99ki, salesTarget99ki2)[1]
monthForecast100ki = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate100ki, endDate100ki, salesTarget100ki)[1]

monthForecast99kitest = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate99ki, endDate99ki, testTarget)[1]


# %%
outputPath = Path("D:/R/SeihanAnalysis/Output/Seasonality")


#Load data table

# monthForecast95ki
# monthForecast96ki
# monthForecast98ki
# monthForecast99ki1
monthForecast99ki2
# monthForecast100ki

#monthForecast99kitest.to_csv(outputPath / "monthForecast99kitest.csv")


# %%
# ax = monthForecast96ki.salesForecast.plot(linewidth = 3, color='tab:blue', title="Month Forecast", label = '96ki (19-20) retail Forecast', legend=True)
# ax = monthForecast96ki.Sales.plot(linewidth = 3, color='tab:purple', label = '96ki (19-20) Retail Actual', legend=True)

ax = monthForecast99ki2.Sales[0:6].plot(linewidth = 3, color='tab:purple', label = '99ki (22-23) Retail Actual', legend=True)
ax = monthForecast99ki1.salesForecast.plot(linewidth = 1, color='tab:orange', title="Month Forecast", label = '99ki (22-23) retail Forecast - 77k', legend=True)
ax = monthForecast99ki2.salesForecast.plot(linewidth = 1, color='tab:blue', title="Month Forecast", label = '99ki (22-23) retail Forecast - 70k', legend=True)

plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'])
#plt.yticks(, ticks)

# plt.xlabel('Gregorian Calendar Months (Apr to Mar)')
plt.ylabel('Sales')

# %%
#Monthly forecast comparison

# ax = monthForecast95ki.salesForecast.plot(linewidth = 3, color='tab:blue', title="Month Forecast", label = '95ki (18-19) retail Forecast', legend=True)
# ax = monthForecast95ki.Sales.plot(linewidth = 3, color='tab:purple', label = '95ki (18-19) Retail Actual', legend=True)

# ax = monthForecast96ki.salesForecast.plot(linewidth = 3, color='tab:blue', title="Month Forecast", label = '96ki (19-20) retail Forecast', legend=True)
# ax = monthForecast96ki.Sales.plot(linewidth = 3, color='tab:purple', label = '96ki (19-20) Retail Actual', legend=True)

ax = monthForecast98ki.salesForecast.plot(linewidth = 3, color='tab:blue', title="Month Forecast", label = '98ki (21-22) Retail Forecast', legend=True)
ax = monthForecast98ki.Sales.plot(linewidth = 3, color='tab:purple', label = '98ki (21-22) Retail Actual', legend=True)

# ax = monthForecast99ki.salesForecast.plot(linewidth = 3, color='tab:blue', title="Month Forecast", label = '99ki (22-23) retail Forecast', legend=True)
# ax = monthForecast99ki.Sales.plot(linewidth = 3, color='tab:purple', label = '99ki (22-23) Retail Actual', legend=True)

plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'])
#plt.yticks(, ticks)

plt.xlabel('Gregorian Calendar Months (Apr to Mar)')
plt.ylabel('Sales')

# %%
ax = monthForecast95ki.salesForecast.plot(linewidth = 2, color='tab:blue', label = '95ki (18-19)', title="Monthly BHL Retail Sales Comparison", legend=True)
# ay = monthForecast95ki['GregorianMonth']
ax = monthForecast96ki.salesForecast.plot(linewidth = 2, color='tab:orange', label = '96ki (19-20)', legend=True)
ax = monthForecast98ki.salesForecast.plot(linewidth = 2, color='tab:green', label = '98ki (21-22)', legend=True)
ax = monthForecast99ki.salesForecast.plot(linewidth = 2, color='tab:purple', label = '99ki (22-23)', legend=True)
ax = monthForecast100ki.salesForecast.plot(linewidth = 2, color='tab:cyan', label = '100ki (23-24)', legend=True)

plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'])
#plt.yticks(, ticks)

plt.xlabel('Gregorian Calendar Months (Apr to Mar)')
plt.ylabel('Sales Forecast')


# %% [markdown]
# #### Day forecast

# %%
#daily forecast
dayForecast95ki = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate95ki, endDate95ki, salesTarget95ki)[0]
dayForecast96ki = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate96ki, endDate96ki, salesTarget96ki)[0]
dayForecast98ki = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate98ki, endDate98ki, salesTarget98ki)[0]
dayForecast99ki = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate99ki, endDate99ki, salesTarget99ki)[0]
dayForecast100ki = gregorianForecast(data_dir, sheet_name, dateCol, salesVar, startDate100ki, endDate100ki, salesTarget100ki)[0]


# %%
#Load table

# dayForecast95ki
dayForecast96ki
# dayForecast98ki
# dayForecast99ki
# dayForecast100ki

# %%
ax = dayForecast96ki.salesForecast.plot(linewidth = 1, color='tab:purple', label = '96ki Daily Retail Sales Forecast', legend=True, title="Day Forecast", )
ax = dayForecast96ki.Sales.plot(linewidth = 1, color='tab:blue', label = '96ki Daily Retail Sales Actual', legend=True)

plt.xlabel('Gregorian Calendar days')
plt.ylabel('Sales Forecast')

# %%
ax = dayForecast19.Sales.plot(linewidth = 1, color='b', label = '2019 Daily Retail Actual', legend=True)
ax = dayForecast19.salesForecast.plot(linewidth = 1, color='c', label = '2019 Daily Retail Forecast', legend=True)


# %%
ax = dayForecast21.salesForecast.plot(linewidth = 1, color='b', label = '2021 Daily Retail Forecast', legend=True)
ax = dayForecast21.Sales.plot(linewidth = 1, color='c', label = '2021 Daily Retail Actual', legend=True)


# %%
ax = dayForecast18.salesForecast.plot(linewidth = 1, color='c', label = '2018 Daily Retail Forecast', legend=True, title="Day Forecast", )
ax = dayForecast19.salesForecast.plot(linewidth = 1, color='b', label = '2019 Daily Retail Forecast', legend=True)
ax = dayForecast21.salesForecast.plot(linewidth = 1, color='r', label = '2021 Daily Retail Forecast', legend=True)


# %%
outputPath = Path("D:/R/SeihanAnalysis/Output/Seasonality")

dayForecast19.to_csv(outputPath / "dayForecast19.csv")
dayForecast20.to_csv(outputPath / "dayForecast20.csv")
