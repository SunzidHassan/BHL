# %% [markdown]
# ## Setup

# %%
#%reset

# %% [markdown]
# ### Load libraries

# %%

from warnings import simplefilter
simplefilter("ignore")  # ignore warnings to clean up output cells

import numpy as np
import pandas as pd
import math as mt

from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_pacf

import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.graph_objects as go

import datetime as dt
import openpyxl
from pathlib import Path


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

# %%
# "Shift + Alt + F" for 'Black' formatting

# %% [markdown]
# ## Input Filtering

# %% [markdown]
# ### Function

# %%
def seasonalInput(
    input_data_dir,
    sheet_name,
    dateCol,
    modelName="TTL",
    modelCode="TTL",
    modelColor="TTL",
    modelSegment="TTL",
    region="TTL",
    dealer="TTL",
    variable="TTL"
):

    # dataload
    populationData = pd.read_excel(
        input_data_dir, sheet_name=sheet_name, parse_dates=[dateCol]
    )

    populationData = populationData.set_index(dateCol).to_period(freq="D")

    # Filter data
    sampleData = populationData.copy()
    sampleData = sampleData.loc[(sampleData.HijriYear != 1338) & (sampleData.HijriYear != 1441) & (sampleData.HijriYear != 1444)]
    sampleData = sampleData[sampleData['Value'].notnull()]
    

    #Filter against given variables
    if modelName == "TTL":
        pass
    elif variable not in set(sampleData.Variable):
        print("Please input a valid variable name)")
        return False
    else:
        sampleData = sampleData.loc[(sampleData.Variable == variable)]
        # Day sum

    
    if modelName == "TTL":
        pass        
    elif modelName not in set(sampleData.Model_Name):
            print("Please input a valid Model Name")
            return False
    else:
        sampleData = sampleData.loc[(sampleData.Model_Name == modelName)]
        # Day sum


    if modelName == "TTL":
        pass        
    elif modelCode not in set(sampleData.Model_Code):
        print("Please input a valid Model Code")
        return False
    else:
        sampleData = sampleData.loc[(sampleData.Model_Code == modelCode)]
        # Day sum

    
    if modelName == "TTL":
        pass        
    elif modelColor not in set(sampleData.Color):
        print("Please input a valid Model Color")
        return False
    else:
        sampleData = sampleData.loc[(sampleData.Color == modelColor)]
        # Day sum

    
    if modelName == "TTL":
        pass        
    elif modelSegment not in set(sampleData.ModelSegment):
        print("Please input a valid model segment")
        return False
    else:
        sampleData = sampleData.loc[(sampleData.ModelSegment == modelSegment)]
        # Day sum
        
    
    if modelName == "TTL":
        pass        
    elif region not in set(sampleData.Region):
        print("Please input a valid Region")
        return False
    else:
        sampleData = sampleData.loc[(sampleData.Region == region)]
        # Day sum

    
    if modelName == "TTL":
        pass        
    elif dealer not in set(sampleData.Dealer):
        print("Please input a valid Dealer")
        return False
    else:
        sampleData = sampleData.loc[(sampleData.Dealer == dealer)]
        # Day sum

    # Todo: daily sum of filtered data    

    populationCount = populationData.Value.count()
    sampleCount = sampleData.Value.count()

    return sampleData, populationCount, sampleCount

# %% [markdown]
# ### Output

# %%
input_data_dir = Path("D:/R/SeihanAnalysis/Input/SalesData/IndependentVarsPy.xlsx")
sheet_name = "GregorianHirjiCalendarSales"
dateCol = "GregorianDate"

seasonalInputData = seasonalInput(
    input_data_dir,
    sheet_name,
    dateCol,
    # modelName="TTL",
    # modelCode="TTL",
    # modelColor="TTL",
    # modelSegment="TTL",
    # region="TTL",
    # dealer="TTL",
    variable="BHLRetail ACT"
)

inputData = seasonalInputData[0]  # type: ignore
populationCount = seasonalInputData[1]  # type: ignore
sampleCount = seasonalInputData[2]  # type: ignore

print("Population count is = ", populationCount, "and Sample count is =", sampleCount)
inputData


# %% [markdown]
# ## Seasonal Function

# %% [markdown]
# ### Function

# %%
def seasonal_weight(sampleData, populationCount, sampleCount, timeFrame, seasonalVar):
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

    # convert function arguments
    timeFrame = sampleData[timeFrame]
    seasonalVar = sampleData[seasonalVar]

    # Pivot Data
    Pivot = sampleData.pivot_table(
        index=timeFrame, columns=seasonalVar, values=["Value"], aggfunc="sum"
    )

    # Small seasonality within defined time frame
    seasonalChange = Pivot.copy()
    seasonalChange.Value = 0
    seasonalChange.Value.iloc[:, 0] = 1  # type: ignore

    for longRange in Pivot.index:
        for shortRange in range(1, len(Pivot.columns)):
            seasonalChange.loc[longRange, ("Value", shortRange + 1)] = 1 + (
                (
                    Pivot.loc[longRange, ("Value", shortRange + 1)]
                    - Pivot.loc[longRange, ("Value", shortRange)]
                )
                / Pivot.loc[longRange, ("Value", shortRange)]
            )

    # Cumulative change
    cumulativeChange = seasonalChange.copy()
    cumulativeChange.Value[2 : len(seasonalChange.columns)] = 0

    for longRange in seasonalChange.index:
        for shortRange in range(1, len(seasonalChange.columns)):
            cumulativeChange.loc[longRange, ("Value", shortRange + 1)] = (
                cumulativeChange.loc[longRange, ("Value", shortRange)]
                * cumulativeChange.loc[longRange, ("Value", shortRange + 1)]
            )

    cumulativeChangeSum = cumulativeChange.sum(axis=1)

    # Same scale
    shortrangeSeasonalWeight = cumulativeChange.copy()
    shortrangeSeasonalWeight.Value[2 : len(seasonalChange.columns)] = 0

    for longRange in seasonalChange.index:
        for shortRange in range(1, len(seasonalChange.columns) + 1):
            shortrangeSeasonalWeight.loc[longRange, ("Value", shortRange)] = (
                len(seasonalChange.columns) / cumulativeChangeSum[longRange]
            ) * cumulativeChange.loc[longRange, ("Value", shortRange)]

    return shortrangeSeasonalWeight

# %% [markdown]
# ### Function output

# %% [markdown]
# #### Seasonality output function common input

# %%
input_data_dir = Path("D:/R/SeihanAnalysis/Input/SalesData/IndependentVarsPy.xlsx")
sheet_name = "GregorianHirjiCalendarSales"
dateCol = "GregorianDate"

seasonalInputData = seasonalInput(
    input_data_dir,
    sheet_name,
    dateCol,
    # modelName="TTL",
    # modelCode="TTL",
    # modelColor="TTL",
    # modelSegment="TTL",
    # region="TTL",
    # dealer="TTL",
    variable="BHLRetail ACT"
)

sampleData = seasonalInputData[0]  # type: ignore
populationCount = seasonalInputData[1]  # type: ignore
sampleCount = seasonalInputData[2]  # type: ignore

print("Population count is = ", populationCount, "and Sample count is =", sampleCount)
inputData

# %% [markdown]
# #### Hijri month seasonality in Hijri year

# %%
timeFrame = "HijriYear"
seasonalVar = "HijriMonth"

weight = seasonal_weight(sampleData, populationCount, sampleCount, timeFrame, seasonalVar)
mean = weight.mean()
stDev = stats.tstd(weight)

#Print table
print(mean)
print(stDev)

#Variance ranged graph
x = mean.index.get_level_values(1)
y = mean

fig, ax = plt.subplots()
ax.plot(x, y, '-',linewidth = 2)
ax.fill_between(x, (y + stDev), (y - stDev), alpha = 0.2)

labels= ["Hijri Month Weight"]
plt.legend(labels)

# ax.fill_between(x, meanStd['avg'] + meanStd['StDev'],  meanStd['avg'] - meanStd['StDev'], alpha=0.2)
ax.plot(x, y, 'o', color='white')

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

# %% [markdown]
# #### Weekday seasonality in Hijri Month

# %%
timeFrame = "HijriMonth"
seasonalVar = "WeekdayNum"

weight = seasonal_weight(sampleData, populationCount, sampleCount, timeFrame, seasonalVar)
mean = weight.mean()
stDev = stats.tstd(weight)

#Print table
print(mean)

#Variance ranged graph
x = mean.index.get_level_values(1)
y = mean

fig, ax = plt.subplots()
ax.plot(x, y, '-',linewidth = 2)
ax.fill_between(x, (y + stDev), (y - stDev), alpha = 0.2)

# ax.fill_between(x, meanStd['avg'] + meanStd['StDev'],  meanStd['avg'] - meanStd['StDev'], alpha=0.2)
ax.plot(x, y, 'o', color='tab:blue')

labels= ["Weekday Forecast"]
plt.legend(labels)

plt.xticks([1, 2, 3, 4, 5, 6, 7], ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])

plt.xlabel("Days in Week (Sunday to Saturday)")
plt.ylabel("Daily weight")

# %% [markdown]
# #### Monthday seasonality in Hijri Months

# %%
timeFrame = "HijriMonth"
seasonalVar = "HijriDayPerMonth"

weight = seasonal_weight(sampleData, populationCount, sampleCount, timeFrame, seasonalVar)


mean9 = weight.Value.iloc[8, :]
stDev9 = stats.tstd(weight.Value.iloc[8, :])

#Variance ranged graph
x9 = mean9.index
y9 = mean9

mean12 = weight.Value.iloc[11, :]
stDev12 = stats.tstd(weight.Value.iloc[11, :])

#Variance ranged graph
x12 = mean12.index
y12 = mean12


fig, ax = plt.subplots()
plt.xlabel("Days in Month")
plt.ylabel("Daily weight")

#Figure of Ramadan
ax.plot(x9, y9, '-',linewidth = 2, label = "Eid-ul-Fitr Month")
ax.fill_between(x9, (y9 + stDev9), (y9 - stDev9), alpha = 0.2)
ax.plot(x9, y9, 'o', color='tab:blue')


#Figure of last month
ax.plot(x12, y12, '-',linewidth = 2, label = 'Eid-Ul-Adha Month')
ax.fill_between(x12, (y12 + stDev12), (y12 - stDev12), alpha = 0.2)
ax.plot(x12, y12, 'o', color='tab:orange')

plt.legend()

# %% [markdown]
# #### Export weights

# %% [markdown]
# ## Hijri seasonality to Gregorian forecast function

# %% [markdown]
# ### Output function for Hijri to Gregorian Function

# %%
#Processing input for Hijri to Gregorian Function
def combined_Weight(input_data_dir, sheet_name, dateCol, modelName="TTL", modelCode="TTL", modelColor="TTL", modelSegment="TTL", region="TTL", dealer="TTL", variable="TTL"):
    seasonalInputData = seasonalInput(
        input_data_dir,
        sheet_name,
        dateCol,
        modelName,
        modelCode,
        modelColor,
        modelSegment,
        region,
        dealer,
        variable
    )

    sampleData = seasonalInputData[0]  # type: ignore
    populationCount = seasonalInputData[1]  # type: ignore
    sampleCount = seasonalInputData[2]  # type: ignore

    #Input for Hijri to Gregorian Seasonality
    dailySales = pd.read_excel(input_data_dir, sheet_name=sheet_name, parse_dates=[dateCol])
    dailySales = dailySales.loc[(dailySales.Variable == variable)]

    # dailySales = dailySales#.set_index(dateCol).to_period(freq='D')

    calenderMergeSource = dailySales.loc[:, (
            dateCol,
            "HijriYear",
            "HijriMonth",
            "HijriDayPerMonth",
            "WeekdayNum",
            "Value"
        ),
    ]

    ##Load seasonality from function and convert list that you received from seasonality function into dataframe - so that it can be merged

    # month per year weight
    hijrimonthPerYearSeasonalWeight = seasonal_weight(sampleData, populationCount, sampleCount, "HijriYear", "HijriMonth").mean()
    hijrimonthPerYearSeasonalWeightDF = pd.DataFrame(hijrimonthPerYearSeasonalWeight, columns=['monthPerYearWeight'])

    # weekday weight
    WeekdayperMonthSeasonalWeight = seasonal_weight(sampleData, populationCount, sampleCount, "HijriMonth", "WeekdayNum").mean()
    WeekdayperMonthSeasonalWeightDF = pd.DataFrame(
        WeekdayperMonthSeasonalWeight, columns=["weekdayPerMonthWeight"]
    )

    # monthday weight
    old = seasonal_weight(
        sampleData, populationCount, sampleCount, "HijriMonth", "HijriDayPerMonth"
    )
    new = pd.DataFrame(
        index=range(361),
        columns=["HijriMonth", "HijriDayPerMonth", "monthdayPerMonthWeight"],
    )

    for i in range(1, 13):
        for j in range(1, 31):
            row = ((i - 1) * 30) + j
            new["HijriMonth"].iloc[row] = i
            new["HijriDayPerMonth"].iloc[row] = j
            new["monthdayPerMonthWeight"].iloc[row] = old.loc[i, ("Value", j)]

    monthdayPerMonthSeasonalWeight = new.iloc[1:, :]

    # join 3 weights with the gregorian calendar
    monthWeightJoin = pd.merge(
        calenderMergeSource.reset_index(),
        hijrimonthPerYearSeasonalWeightDF,
        on="HijriMonth",
        how="left",
    )

    weekdayWeightJoin = pd.merge(
        monthWeightJoin, WeekdayperMonthSeasonalWeightDF, on="WeekdayNum", how="left"
    )

    monthdayWeightJoin = pd.merge(
        weekdayWeightJoin,
        monthdayPerMonthSeasonalWeight,
        on=["HijriMonth", "HijriDayPerMonth"],
        how="left",
    )

    combinedWeight = monthdayWeightJoin.copy()

    return combinedWeight, calenderMergeSource

# %% [markdown]
# ### Hijri to Gregorian Function

# %%
def gregorianForecast(
    weightInput, startDate, endDate, salesTarget
):

    combinedWeight = weightInput[0]
    calenderMergeSource = weightInput[1]

    monthWeight = combinedWeight.loc[
        (combinedWeight.GregorianDate.dt.date > startDate)
        & (combinedWeight.GregorianDate.dt.date < endDate)
    ][["GregorianDate", "HijriYear", "HijriMonth", "monthPerYearWeight"]]

    monthWeightPivot = monthWeight.pivot_table(
        index=(monthWeight.HijriYear, monthWeight.HijriMonth),
        values=["monthPerYearWeight"],  # type: ignore
        aggfunc=("mean", "count"),
    )

    # new monthday weight
    monthdayWeight = combinedWeight.loc[
        (combinedWeight.GregorianDate.dt.date > startDate)
        & (combinedWeight.GregorianDate.dt.date < endDate)
    ][
        [
            "HijriYear",
            "HijriMonth",
            "HijriDayPerMonth",
            "monthdayPerMonthWeight",
            "weekdayPerMonthWeight",
        ]
    ]

    # monthdayWeightPivot = monthdayWeight.pivot_table(index = (monthdayWeight.HijriYear, monthdayWeight.HijriMonth, monthdayWeight.HijriDayPerMonth), values=["monthdayPerMonthWeight", "weekdayPerMonthWeight"], aggfunc = ('mean'))
    monthdayWeightPivot1 = monthdayWeight.pivot_table(
        index=(
            monthdayWeight.HijriYear,
            monthdayWeight.HijriMonth,
            monthdayWeight.HijriDayPerMonth,
        ),
        values=["monthdayPerMonthWeight"],  # type: ignore
        aggfunc="mean",
    )
    monthdayWeightPivot2 = monthdayWeight.pivot_table(
        index=(
            monthdayWeight.HijriYear,
            monthdayWeight.HijriMonth,
            monthdayWeight.HijriDayPerMonth,
        ),
        values=["weekdayPerMonthWeight"],  # type: ignore
        aggfunc="mean",
    )
    monthdayWeightPivot = pd.merge(
        monthdayWeightPivot1,
        monthdayWeightPivot2,
        on=("HijriYear", "HijriMonth", "HijriDayPerMonth"),
        how="left",
    )

    # calculate new month day weight by multiplying monthday and weekday weights
    for years in monthWeight.HijriYear.unique():
        for months in monthdayWeightPivot.loc[years].index.get_level_values(0).unique():
            for monthDays in monthdayWeightPivot.loc[years, months].index:
                monthdayWeightPivot.loc[
                    (years, months, monthDays), "newMonthdayWeight"
                ] = (monthdayWeightPivot.loc[
                        (years, months, monthDays), "monthdayPerMonthWeight"
                    ]
                ) * (
                    monthdayWeightPivot.loc[
                        (years, months, monthDays), "weekdayPerMonthWeight"
                    ]
                )

    # calculate total monthday weight of each month
    for years in monthWeight.HijriYear.unique():
        for months in monthdayWeightPivot.loc[years].index.get_level_values(0).unique():
            monthdayWeightPivot.loc[
                (years, months), "monthWeightTotal"
            ] = monthdayWeightPivot.loc[(years, months), "newMonthdayWeight"].sum()

    # keep total monthday weight of each month for the filtered timeline
    totalFilteredMonthdayWeight = monthdayWeightPivot.reset_index(level=2, drop=True)[
        ["monthWeightTotal"]
    ].drop_duplicates(keep="first")

    # calculating total possible sum of monthday weights in a month, and sum of monthday weight for filtered days
    tempCombinedWeight = combinedWeight[
        [
            "HijriYear",
            "HijriMonth",
            "HijriDayPerMonth",
            "weekdayPerMonthWeight",
            "monthdayPerMonthWeight",
        ]
    ]
    tempCombinedWeight["newDailyWeight"] = (
        tempCombinedWeight.weekdayPerMonthWeight
        * tempCombinedWeight.monthdayPerMonthWeight
    )
    totalMonthdayWeight = tempCombinedWeight.pivot_table(
        index=("HijriYear", "HijriMonth"), values=["newDailyWeight"], aggfunc=("sum")  # type: ignore
    )

    # new month weights for given total weight scale. If month = 12 and day = 30, then new weight = old weight
    for years in monthWeight.HijriYear.unique():
        for months in monthWeightPivot.loc[years].index:
            monthWeightPivot.loc[(years, months), ("monthPerYearWeight", "newMean")] = monthWeightPivot.loc[
                (years, months), ("monthPerYearWeight", "mean")
            ] * ((totalFilteredMonthdayWeight.loc[(years, months), "monthWeightTotal"])
                / (totalMonthdayWeight.loc[(years, months), "newDailyWeight"]))

    # calculate sum of all month weight for the given timeline
    totalNewMonthlyWeight = monthWeightPivot.loc[
        :, ("monthPerYearWeight", "newMean")
    ].sum()

    # calculate monthly sales from total sales forecast, total month weight and individual monthly weights
    for years in monthWeight.HijriYear.unique():
        for months in monthWeightPivot.loc[years].index:
            monthWeightPivot.loc[
                (years, months), ("monthPerYearWeight", "monthSales")] = round(salesTarget
                * (monthWeightPivot.loc[
                        (years, months), ("monthPerYearWeight", "newMean")]
                    / totalNewMonthlyWeight),
                0,)

    # forecasting daily sales from monthly sales forecast and daily weights
    for years in monthWeight.HijriYear.unique():
        for months in monthdayWeightPivot.loc[years].index.get_level_values(0).unique():
            for monthDays in monthdayWeightPivot.loc[years, months].index:
                monthdayWeightPivot.loc[
                    (years, months, monthDays), "salesForecast"
                ] = round(((monthWeightPivot.loc[
                                (years, months), ("monthPerYearWeight", "monthSales")])
                        / (monthdayWeightPivot.loc[(years, months, monthDays), "monthWeightTotal"]))
                    * (monthdayWeightPivot.loc[(years, months, monthDays), "newMonthdayWeight"]),  # type: ignore
                    0,)

    # Gregorian sum
    targetCalender = calenderMergeSource.reset_index()
    targetCalender = targetCalender.loc[
        (targetCalender.GregorianDate.dt.date > startDate)
        & (targetCalender.GregorianDate.dt.date < endDate)
    ]

    targetGregorianCalendarSum = pd.merge(
        targetCalender,
        monthdayWeightPivot.loc[:, "salesForecast"].reset_index(),
        on=["HijriYear", "HijriMonth", "HijriDayPerMonth"],
        how="left",
    )

    targetGregorianCalendarSum["GregorianYear"] = targetGregorianCalendarSum.GregorianDate.dt.year
    
    targetGregorianCalendarSum["GregorianMonth"] = targetGregorianCalendarSum.GregorianDate.dt.month

    targetGregorianCalendarMonthlySumSales = targetGregorianCalendarSum.pivot_table(
        index=("GregorianYear", "GregorianMonth"), values=["Value"], aggfunc="sum"  # type: ignore
    )
    
    targetGregorianCalendarMonthlySumForecast = targetGregorianCalendarSum.pivot_table(
        index=("GregorianYear", "GregorianMonth"),
        values=["salesForecast"],  # type: ignore
        aggfunc="sum",
    )
    
    targetGregorianCalendarMonthlySum = pd.merge(
        targetGregorianCalendarMonthlySumSales.reset_index(),
        targetGregorianCalendarMonthlySumForecast.reset_index(),
        on=("GregorianYear", "GregorianMonth"),
        how="left",
    )

    # targetGregorianCalendarMonthlySum = targetGregorianCalendarMonthlySum
    # targetGregorianCalendarMonthlySum['GregorianDate'] = dt.date(targetGregorianCalendarMonthlySum.GregorianYear, targetGregorianCalendarMonthlySum.GregorianMonth, 1)

    return targetGregorianCalendarSum, targetGregorianCalendarMonthlySum

# %% [markdown]
# ### Output

# %% [markdown]
# #### Common Function Input

# %%
# Parameters

input_data_dir = Path("D:/R/SeihanAnalysis/Input/SalesData/IndependentVarsPy.xlsx")
sheet_name = "GregorianHirjiCalendarSales"
dateCol = "GregorianDate"
variable = "BHLRetail ACT"
# modelName="TTL",
# modelCode="TTL",
# modelColor="TTL",
# region="TTL",
# dealer="TTL",

weightInput = combined_Weight(input_data_dir = input_data_dir, sheet_name = sheet_name, dateCol = dateCol, variable = variable)

startDate95ki = dt.date(2018, 4, 1)
endDate95ki = dt.date(2019, 3, 31)

startDate96ki = dt.date(2019, 4, 1)
endDate96ki = dt.date(2020, 3, 31)

startDate97ki = dt.date(2020, 4, 1)
endDate97ki = dt.date(2021, 3, 31)

startDate98ki = dt.date(2021, 4, 1)
endDate98ki = dt.date(2022, 3, 31)

startDate99ki = dt.date(2022, 4, 1)
endDate99ki = dt.date(2023, 3, 31)

startDate100ki = dt.date(2023, 4, 1)
endDate100ki = dt.date(2024, 3, 31)


salesTarget95ki = 47626
salesTarget96ki = 58959
salesTarget98ki = 74903
salesTarget99ki77 = 77200
salesTarget99ki68 = 68800
salesTarget99ki60 = 60020
salesTarget100ki12 = 12
salesTarget100ki85 = 85000
salesTarget100ki100 = 100000
testTarget = 1000

# %% [markdown]
# #### Month forecast

# %%
# Monthly forecast comparison
monthForecast95ki = gregorianForecast(
    weightInput, startDate95ki, endDate95ki, salesTarget95ki)[1]

monthForecast96ki = gregorianForecast(
    weightInput, startDate96ki, endDate96ki, salesTarget96ki)[1]

monthForecast98ki = gregorianForecast(
    weightInput, startDate98ki, endDate98ki, salesTarget98ki)[1]

monthForecast99kitest = gregorianForecast(
    weightInput, startDate99ki, endDate99ki, testTarget)[1]
    
monthForecast99ki60 = gregorianForecast(
    weightInput, startDate99ki, endDate99ki, salesTarget99ki60)[1]

monthForecast99ki68 = gregorianForecast(
    weightInput, startDate99ki, endDate99ki, salesTarget99ki68)[1]

monthForecast99ki77 = gregorianForecast(
    weightInput, startDate99ki, endDate99ki, salesTarget99ki77)[1]

salesTarget100ki85 = gregorianForecast(
    weightInput, startDate100ki, endDate100ki, salesTarget100ki85)[1]

salesTarget100ki12 = gregorianForecast(
    weightInput, startDate100ki, endDate100ki, salesTarget100ki12)[1]

salesTarget100ki100 = gregorianForecast(
    weightInput, startDate100ki, endDate100ki, salesTarget100ki100)[1]

#Load data table

# monthForecast95ki
# monthForecast96ki
# monthForecast98ki
# monthForecast99ki1
# monthForecast99ki2
# monthForecast100ki


# %%
# ax = monthForecast96ki.salesForecast.plot(linewidth = 3, color='tab:blue', title="Month Forecast", label = '96ki (19-20) retail Forecast', legend=True)
# ax = monthForecast96ki.Sales.plot(linewidth = 3, color='tab:purple', label = '96ki (19-20) Retail Actual', legend=True)

ax = monthForecast99ki68.Value[0:7].plot(
    linewidth=3,
    color="tab:purple",
    label="99ki (22-23) Retail Actual",
    legend=True
)

ax = monthForecast99ki60.salesForecast.plot(
    linewidth=1,
    color="tab:pink",
    title="Month Forecast",
    label="99ki (22-23) Retail Forecast - 60k",
    legend=True,
)
ax = monthForecast99ki68.salesForecast.plot(
    linewidth=1,
    color="tab:orange",
    title="Month Forecast",
    label="99ki (22-23) Retail Forecast - 68k",
    legend=True,
)

ax = monthForecast99ki77.salesForecast.plot(
    linewidth=1,
    color="tab:blue",
    title="Month Forecast",
    label="99ki (22-23) Retail Forecast - 70k",
    legend=True,
)

plt.xticks(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"])

# plt.xlabel('Gregorian Calendar Months (Apr to Mar)')
plt.ylabel("Sales")

# %% [markdown]
# #### Day forecast

# %%
# daily forecast
dayForecast95ki = gregorianForecast(
    weightInput,
    startDate95ki,
    endDate95ki,
    salesTarget95ki
)[0]

dayForecast96ki = gregorianForecast(
    weightInput,
    startDate96ki,
    endDate96ki,
    salesTarget96ki
)[0]

dayForecast98ki = gregorianForecast(
    weightInput,
    startDate98ki,
    endDate98ki,
    salesTarget98ki
)[0]

dayForecast99ki68 = gregorianForecast(
    weightInput,
    startDate99ki,
    endDate99ki,
    salesTarget99ki68
)[0]

# dayForecast100ki = gregorianForecast(
#     weightInput,
#     startDate100ki,
#     endDate100ki,
#     salesTarget100ki,
# )[0]

#Load table

# dayForecast95ki
# dayForecast96ki
# dayForecast98ki
# dayForecast99ki
# dayForecast100ki

# %%
ax = dayForecast99ki70.salesForecast.plot(
    linewidth=1,
    color="tab:purple",
    label="96ki Daily Retail Sales Forecast",
    legend=True,
    title="Day Forecast",
)
ax = dayForecast99ki70.Value.plot(
    linewidth=1, color="tab:blue", label="96ki Daily Retail Sales Actual", legend=True
)

plt.xlabel("Gregorian Calendar days")
plt.ylabel("Sales Forecast")

# %% [markdown]
# ## Value chain optimization

# %% [markdown]
# ### Input and manual code

# %%
optimiserPath = Path("D:/R/SeihanAnalysis/Input/Optimiser")

valueChainData = pd.read_excel(optimiserPath / "OptimizerTestInput.xlsx", parse_dates=["Date"]).loc[:, ['Date', 'CBUDemand']]
#valueChainData = valueChainData.set_index('Date').to_period(freq="M")

lotSize = 100

valueChainData['CBUDemandLot'] = np.divide(valueChainData['CBUDemand'], lotSize)
valueChainData['CBUInventoryQty'] = 0.
valueChainData['CBUStockDaysReq'] = 5.
valueChainData['CBUInventoryLot'] = 0.
valueChainData['ProductionQty'] = 0.
valueChainData['ProductionLot'] = 0.
valueChainData['ProductionUpperBound'] = 15.
valueChainData['ProductionLowerBound'] = 0.
valueChainData['CKDInventoryQty'] = 0.
valueChainData['CKDInventoryLot'] = 0.
valueChainData['CKDStockDaysReq'] = 25.
valueChainData['BHLInQty'] = 0.
valueChainData['BHLInLot'] = 0.
valueChainData['OrderQty'] = 0.
valueChainData['OrderLot'] = 0.
valueChainData['ProductionLot'] = 0.

CBUDemandLot = valueChainData.loc[:, 'CBUDemandLot'].values
CBUInventoryQty = valueChainData.loc[:, 'CBUInventoryQty'].values
CBUStockDaysReq = valueChainData.loc[:, 'CBUStockDaysReq'].values
CBUInventoryLot = valueChainData.loc[:, 'CBUInventoryLot'].values
ProductionQty = valueChainData.loc[:, 'ProductionQty'].values
ProductionLot = valueChainData.loc[:, 'ProductionLot'].values
ProductionUpperBound = valueChainData.loc[:, 'ProductionUpperBound'].values
ProductionLowerBound = valueChainData.loc[:, 'ProductionLowerBound'].values
CKDInventoryQty = valueChainData.loc[:, 'CKDInventoryQty'].values
CKDInventoryLot = valueChainData.loc[:, 'CKDInventoryLot'].values
CKDStockDaysReq = valueChainData.loc[:, 'CKDStockDaysReq'].values
BHLInQty = valueChainData['BHLInQty'].values
BHLInLot = valueChainData['BHLInLot'].values
OrderQty = valueChainData.loc[:, 'OrderQty'].values
OrderLot = valueChainData.loc[:, 'OrderLot'].values
ProductionLot = valueChainData.loc[:, 'ProductionLot'].values


# %%


# simply minimize the total production
prod_objective_fun = lambda ProductionLot: ProductionLot[i]

#bounds = [(ProductionLowerBound, ProductionUpperBound)]
prod_bounds = [(0., 10.)] * (len(valueChainData))
#bounds = [(0., 10.)]

# production in month i is greater than/equal to (-starting inventory + req demand + ending inventory req)
prod_constraint = [{'type': 'ineq', 'fun': lambda ProductionLot: (ProductionLot[i]) - (- CBUInventoryLot[i] + CBUDemandLot[i] + (CBUDemandLot[i+1]*(CBUStockDaysReq[i]/30)))}]

prod_result = minimize(prod_objective_fun, # objective function
                ProductionLot,   # output
                bounds=prod_bounds,  # bounds of solution
                constraints=prod_constraint) # constraints

ProductionLot[i] = mt.ceil(prod_result.fun)   # take ceiling value for fraction lot answer
CBUInventoryLot[i+1] = ProductionLot[i] + CBUInventoryLot[i] - CBUDemandLot[i]   # CBU inventory given full lot

## CKD optimization

bhlIn_objective_fun = lambda BHLInLot: BHLInLot[i]
# order_bounds = 

bhlIn_constraint = [{'type': 'ineq', 'fun': lambda BHLInLot: (BHLInLot[i]) - (- CKDInventoryLot[i] + ProductionLot[i] + (ProductionLot[i+1]*(CKDStockDaysReq[i]/30)))}]

bhlIn_bounds = [(0., 40.)] * (len(valueChainData))

bhlIn_result = minimize(bhlIn_objective_fun, # objective function
                BHLInLot,   # output
                bounds=bhlIn_bounds,  # bounds of solution
                constraints=bhlIn_constraint) # constraints


BHLInLot[i] = mt.ceil(bhlIn_result.fun)   # take ceiling value for fraction lot answer
CKDInventoryLot[i+1] = BHLInLot[i] + CKDInventoryLot[i] - ProductionLot[i]   # CBU inventory given full lot

## todo: order to bhl in

# order_constraint = [{'type': 'ineq', 'fun': lambda OrderLot: np.ceil((0.4*OrderLot[i+2]))  + (CKDInventoryLot[i+3]-(np.ceil(0.4*CKDInventoryLot[i+3])))
# (OrderLot[i]) - (- CBUInventoryLot[i] + CBUDemandLot[i] + (CBUDemandLot[i+1]*(CBUStockDaysReq[i]/30)))}]

# for ckd without any extra days:
for i in range(len(BHLInLot)-3):
OrderLot[i] = np.ceil((0.4*BHLInLot[i+2]))  + (BHLInLot[i+3]-(np.ceil(0.4*BHLInLot[i+3])))


valueChainData['ProductionLot'] = ProductionLot
valueChainData['CBUInventoryLot'] = CBUInventoryLot
valueChainData['CKDInventoryLot'] = CKDInventoryLot
valueChainData['BHLInLot'] = BHLInLot
valueChainData['OrderLot'] = OrderLot

valueChainData['CBUInventoryQty'] = valueChainData['CBUInventoryLot'] * lotSize
valueChainData['ProductionQty'] = valueChainData['ProductionLot'] * lotSize
valueChainData['CKDInventoryQty'] = valueChainData['CKDInventoryLot'] * lotSize
valueChainData['BHLInQty'] = valueChainData['BHLInLot'] * lotSize
valueChainData['OrderQty'] = valueChainData['OrderLot'] * lotSize


# # different approach of CKD stock requirement - keep 1 lot extra at all times?
# OrderLot[1] = OrderLot[1] + 1
# OrderLot[-1] = OrderLot[-1] - 1


valueChainData

# %%
valueChainOutputPath = Path("D:/R/SeihanAnalysis/Output/Seasonality")
valueChainData.to_csv(valueChainOutputPath / "seasonalityValueChain.csv")


# %% [markdown]
# ### For multiple model-color

# %%


# %% [markdown]
# ### Function and Output

# %%
# WOrking code without CKD part

for i in range(len(valueChainData)-1):

    # simply minimize the total production
    objective_fun = lambda ProductionLot: ProductionLot[i]

    #bounds = [(ProductionLowerBound, ProductionUpperBound)]
    bounds = [(0., 10.)] * (len(valueChainData))
    #bounds = [(0., 10.)]

    # production in month i is greater than/equal to (-starting inventory + req demand + ending inventory req)
    constraint = [{'type': 'ineq', 'fun': lambda ProductionLot: (ProductionLot[i]) - (- CBUInventoryLot[i] + CBUDemandLot[i] + (CBUDemandLot[i+1]*(CBUStockDaysReq[i]/30)))}]
    
    result = minimize(objective_fun, # objective function
                    ProductionLot,   # output
                    bounds=bounds,  # bounds of solution
                    constraints=constraint) # constraints
    
    ProductionLot[i] = mt.ceil(result.fun)   # take ceiling value for fraction lot answer
    CBUInventoryLot[i+1] = ProductionLot[i] + CBUInventoryLot[i] - CBUDemandLot[i]   # CBU inventory given full lot

valueChainData['ProductionLot'] = ProductionLot
valueChainData['CBUInventoryLot'] = CBUInventoryLot

valueChainData['ProductionQty'] = valueChainData['ProductionLot'] * lotSize
valueChainData['CBUInventoryQty'] = valueChainData['CBUInventoryLot'] * lotSize

valueChainData

# %%
#copied working function from optimizer without dataframe
def productionPlanner(CBUDemand, CBUInventory, lotSize, cbu_inv_req, bounds):
    prodLot = [0.] * (len(CBUDemand.Demand)-1) #create an holder array
    
    # demand and CBU inventory lot size fractions
    demandLot = np.divide(CBUDemand.Demand, lotSize)
    CBUInvLot = np.divide(CBUInventory, lotSize)
    

    for i in range(len(CBUDemand.Demand)-1):

        # simply minimize the total production
        objective_fun = lambda prodLot: prodLot[i]

        # production in month i is greater than/equal to (-starting inventory + req demand + ending inventory req)
        constraint = [{'type': 'ineq', 'fun': lambda prodLot: (prodLot[i]) - (- CBUInvLot[i] + demandLot[i] + demandLot[i+1]*cbu_inv_req[i+1])}]
        
        result = minimize(objective_fun, # objective function
                        prodLot,   # output
                        bounds=bounds,  # bounds of solution
                        constraints=constraint) # constraints
        
        prodLot[i] = mt.ceil(result.fun)   # take ceiling value for fraction lot answer
        CBUInvLot[i+1] = prodLot[i] + CBUInvLot[i] - demandLot[i]   # CBU inventory given full lot
    
    ckdLot = prodLot.copy()
    orderLot = np.array([0.] * (len(ckdLot)+3))
    production = np.multiply(prodLot, lotSize)  # lotsize to production quantity
    ckd = np.multiply(ckdLot, lotSize)
    CBUInv = np.multiply(CBUInvLot, lotSize)   # CBU lot size to quantity

    for i in range(len(prodLot)-3):
        orderLot[i] = np.ceil((0.4*ckdLot[i+2])) +  + (ckdLot[i+3]-(np.ceil(0.4*ckdLot[i+3])))   # CBU requirement to order
    order = np.multiply(orderLot, lotSize) # order lot size to quantity

    return CBUDemand, production, CBUInv, ckd, order

# %%
##### Discarded: direct dataframe optimization

for i in range(len(valueChainData)-1):

    # simply minimize the total production
    # old: objective_fun = lambda prod: prod[i]
    objective_fun = lambda ProductionLot: ProductionLot[i]

    #bound
    bounds = [(valueChainData.loc[i, 'ProductionUpperBound'], valueChainData.loc[i, 'ProductionLowerBound'])]
    
    # production in month i is greater than/equal to (-starting inventory + req demand + ending inventory req)
    # old: constraint = [{'type': 'ineq', 'fun': lambda prodLot: (prodLot[i]) - (- CBUInvLot[i] + demandLot[i] + demandLot[i+1]*cbu_inv_req[i+1])}]
    constraint = [{'type': 'ineq', 'fun': lambda valueChainData: (valueChainData.loc[i, 'ProductionLot']) - (- valueChainData.loc[i, 'CBUInventoryLot'] + valueChainData[i, 'CBUDemandLot'] + valueChainData[i+1, 'CBUDemandLot']*(valueChainData[i+1, 'CBUStockDaysReq']/30))}]
    
    result = minimize(objective_fun, # objective function
                    valueChainData,   # output
                    bounds=bounds,  # bounds of solution
                    constraints=constraint) # constraints
    
    # take ceiling value for fraction lot answer
    # prodLot[i] = mt.ceil(result.fun)
    
    #valueChainData.loc[i, 'ProductionLot'] = mt.ceil(valueChainData.loc[i, 'ProductionLot'])

    # CBU inventory given full lot
    # old: CBUInvLot[i+1] = prodLot[i] + CBUInvLot[i] - demandLot[i]
    
    #valueChainData.loc[i+1, 'CBUInventoryLot'] = valueChainData.loc[i, 'ProductionLot'] + alueChainData.loc[i, 'CBUInventoryLot'] - valueChainData[i, 'CBUDemandLot']
