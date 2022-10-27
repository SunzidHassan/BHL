# %% [markdown]
# ## Setup

# %%
%reset

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
    region="TTL",
    dealer="TTL",
    variable="TTL"
):

    # dataload
    populationData = pd.read_excel(
        input_data_dir, sheet_name=sheet_name, parse_dates=[dateCol]
    )

    populationData = populationData.set_index(dateCol).to_period(freq="D")  # type: ignore

    # Filter data
    sampleData = populationData.copy()
    sampleData = sampleData.loc[(sampleData.HijriYear != 1338) & (sampleData.HijriYear != 1441) & (sampleData.HijriYear != 1444)]
    sampleData = sampleData[sampleData['Value'].notnull()]
    

    #Filter against given variables
    if variable not in set(sampleData.Variable):
        print("Please input a valid variable name)")
        return False
    else:
        sampleData = sampleData.loc[(sampleData.Variable == variable)]

    if modelName not in set(sampleData.Model_Name):
            print("Please input a valid Model Name")
            return False
    else:
        sampleData = sampleData.loc[(sampleData.Model_Name == modelName)]
        

    if modelCode not in set(sampleData.Model_Code):
        print("Please input a valid Model Code")
        return False
    else:
        sampleData = sampleData.loc[(sampleData.Model_Code == modelCode)]

    if modelColor not in set(sampleData.Color):
        print("Please input a valid Model Color")
        return False
    else:
        sampleData = sampleData.loc[(sampleData.Color == modelColor)]

    if region not in set(sampleData.Region):
        print("Please input a valid Region")
        return False
    else:
        sampleData = sampleData.loc[(sampleData.Region == region)]

    if dealer not in set(sampleData.Dealer):
        print("Please input a valid Dealer")
        return False
    else:
        sampleData = sampleData.loc[(sampleData.Dealer == dealer)]
    

    populationCount = populationData.Value.count()
    sampleCount = sampleData.Value.count()

    return sampleData, populationCount, sampleCount

# %% [markdown]
# ### Output

# %%
input_data_dir = Path("D:/R/SeihanAnalysis/Input/SalesData/IndependentVarsPy.xlsx")
sheet_name = "GregorianHirjiCalendarSales"
dateCol = "GregorianDate"

# modelName="TTL",
# modelCode="TTL",
# modelColor="TTL",
# region="TTL",
# dealer="TTL",

seasonalInputData = seasonalInput(
    input_data_dir,
    sheet_name,
    dateCol,
    variable="BHLRetail",
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

# modelName="TTL",
# modelCode="TTL",
# modelColor="TTL",
# region="TTL",
# dealer="TTL",

seasonalInputData = seasonalInput(
    input_data_dir,
    sheet_name,
    dateCol,
    variable="BHLRetail",
)

sampleData = seasonalInputData[0]  # type: ignore
populationCount = seasonalInputData[1]  # type: ignore
sampleCount = seasonalInputData[2]  # type: ignore


# %% [markdown]
# #### Hijri month seasonality in Hijri year

# %%
timeFrame = "HijriYear"
seasonalVar = "HijriMonth"

hijrimonthPerYearSeasonalWeight = seasonal_weight(sampleData, populationCount, sampleCount, timeFrame, seasonalVar).mean()

#Print table
print(hijrimonthPerYearSeasonalWeight)

#Load graph
ax = hijrimonthPerYearSeasonalWeight.Value.plot(
    linewidth=3, color="0.5", title="Hijri Month Seasonality"
)

# %% [markdown]
# #### Weekday seasonality in Hijri Month

# %%
timeFrame = "HijriMonth"
seasonalVar = "WeekdayNum"

# Seasonal function
WeekdayperMonthSeasonalWeight = seasonal_weight(
    sampleData, populationCount, sampleCount, timeFrame, seasonalVar
).mean()

#Print table
print(WeekdayperMonthSeasonalWeight)

# Plot weights
ax = WeekdayperMonthSeasonalWeight.Value.plot(
    linewidth=3, color="0.5", title="Weekday seasonality"
)

plt.xticks([1, 2, 3, 4, 5, 6, 7], ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])

plt.xlabel("Days in Week (Sunday to Saturday)")
plt.ylabel("Daily weight")

# %% [markdown]
# #### Monthday seasonality in Hijri Months

# %%
timeFrame = "HijriMonth"
seasonalVar = "HijriDayPerMonth"

# Seasonal function
monthdayPerMonthSeasonalWeight = seasonal_weight(
    sampleData, populationCount, sampleCount, timeFrame, seasonalVar
)

print(monthdayPerMonthSeasonalWeight)

# Plot weights
ax = monthdayPerMonthSeasonalWeight.Value.iloc[8, :].plot(
    linewidth=1,
    color="g",
    label="Ramadan Month (Eid-ul-Fitr Sales)",
    legend=True,
    title="Hijri Month-day Seasonality",
)
ax = monthdayPerMonthSeasonalWeight.Value.iloc[11, :].plot(linewidth = 1, color='b', label = 'Dhu al-Hijja Month (Eid-ul-Adha Sales)', legend=True)

plt.xlabel("Days in Month")
plt.ylabel("Daily weight")

# %% [markdown]
# #### Export weights

# %%
# Export month weight

pd.DataFrame(monthdayPerMonthSeasonalWeight.Sales.iloc[11, :]).to_csv(
    "D:/R/SeihanAnalysis/Output/Seasonality/tempMonth5.csv"
)

# %% [markdown]
# ## Hijri seasonality to Gregorian forecast function

# %% [markdown]
# ### Output function for Hijri to Gregorian Function

# %%
#Processing input for Hijri to Gregorian Function
def combined_Weight(input_data_dir, sheet_name, dateCol, variable, modelName="TTL", modelCode="TTL", modelColor="TTL", region="TTL", dealer="TTL"):
    
    seasonalInputData = seasonalInput(
        input_data_dir,
        sheet_name,
        dateCol,
        modelName,
        modelCode,
        modelColor,
        region,
        dealer,
        variable)


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
# ### Todo: Standard deviation, Range prediction and Fan chart

# %%


# %% [markdown]
# ### Output

# %% [markdown]
# #### Common Function Input

# %%
# Parameters

input_data_dir = Path("D:/R/SeihanAnalysis/Input/SalesData/IndependentVarsPy.xlsx")
sheet_name = "GregorianHirjiCalendarSales"
dateCol = "GregorianDate"
variable = "BHLRetail"
# modelName="TTL",
# modelCode="TTL",
# modelColor="TTL",
# region="TTL",
# dealer="TTL",

weightInput = combined_Weight(input_data_dir, sheet_name, dateCol, variable)

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
salesTarget99ki70 = 70020
salesTarget100ki = 99500
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
    
monthForecast99ki70 = gregorianForecast(
    weightInput, startDate99ki, endDate99ki, salesTarget99ki70)[1]

monthForecast99ki77 = gregorianForecast(
    weightInput, startDate99ki, endDate99ki, salesTarget99ki77)[1]


monthForecast100ki = gregorianForecast(
    weightInput, startDate99ki, endDate99ki, testTarget)[1]


#Load data table

# monthForecast95ki
# monthForecast96ki
# monthForecast98ki
# monthForecast99ki1
# monthForecast99ki2
# monthForecast100ki


# %%
# outputPath = Path("D:/R/SeihanAnalysis/Output/Seasonality")
# monthForecast99kitest.to_csv(outputPath / "monthForecast99kitest.csv")

# %%
# ax = monthForecast96ki.salesForecast.plot(linewidth = 3, color='tab:blue', title="Month Forecast", label = '96ki (19-20) retail Forecast', legend=True)
# ax = monthForecast96ki.Sales.plot(linewidth = 3, color='tab:purple', label = '96ki (19-20) Retail Actual', legend=True)

ax = monthForecast99ki70.Value[0:7].plot(
    linewidth=3,
    color="tab:purple",
    label="99ki (22-23) Retail Actual",
    legend=True
)

ax = monthForecast99ki70.salesForecast.plot(
    linewidth=1,
    color="tab:orange",
    title="Month Forecast",
    label="99ki (22-23) Retail Forecast - 77k",
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

# %%
# Monthly forecast comparison

# ax = monthForecast95ki.salesForecast.plot(linewidth = 3, color='tab:blue', title="Month Forecast", label = '95ki (18-19) retail Forecast', legend=True)
# ax = monthForecast95ki.Sales.plot(linewidth = 3, color='tab:purple', label = '95ki (18-19) Retail Actual', legend=True)

# ax = monthForecast96ki.salesForecast.plot(linewidth = 3, color='tab:blue', title="Month Forecast", label = '96ki (19-20) retail Forecast', legend=True)
# ax = monthForecast96ki.Sales.plot(linewidth = 3, color='tab:purple', label = '96ki (19-20) Retail Actual', legend=True)

ax = monthForecast98ki.salesForecast.plot(
    linewidth=3,
    color="tab:blue",
    title="Month Forecast",
    label="98ki (21-22) Retail Forecast",
    legend=True,
)
ax = monthForecast98ki.Value.plot(
    linewidth=3, color="tab:purple", label="98ki (21-22) Retail Actual", legend=True
)

# ax = monthForecast99ki.salesForecast.plot(linewidth = 3, color='tab:blue', title="Month Forecast", label = '99ki (22-23) retail Forecast', legend=True)
# ax = monthForecast99ki.Sales.plot(linewidth = 3, color='tab:purple', label = '99ki (22-23) Retail Actual', legend=True)

plt.xticks(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"])

# plt.yticks(, ticks)

plt.xlabel("Gregorian Calendar Months (Apr to Mar)")
plt.ylabel("Sales")

# %%
ax = monthForecast95ki.salesForecast.plot(
    linewidth=2,
    color="tab:blue",
    label="95ki (18-19)",
    title="Monthly BHL Retail Sales Comparison",
    legend=True,
)

ax = monthForecast96ki.salesForecast.plot(
    linewidth=2,
    color="tab:orange",
    label="96ki (19-20)",
    legend=True
)

ax = monthForecast98ki.salesForecast.plot(
    linewidth=2,
    color="tab:green",
    label="98ki (21-22)",
    legend=True
)

ax = monthForecast99ki70.salesForecast.plot(
    linewidth=2,
    color="tab:purple",
    label="99ki (22-23)",
    legend=True
)

ax = monthForecast100ki.salesForecast.plot(
    linewidth=2,
    color="tab:cyan",
    label="100ki (23-24)",
    legend=True
)

plt.xticks(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"])

# plt.yticks(, ticks)

plt.xlabel("Gregorian Calendar Months (Apr to Mar)")
plt.ylabel("Sales Forecast")

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

dayForecast99ki70 = gregorianForecast(
    weightInput,
    startDate99ki,
    endDate99ki,
    salesTarget99ki70
)[0]

dayForecast100ki = gregorianForecast(
    weightInput,
    startDate100ki,
    endDate100ki,
    salesTarget100ki,
)[0]

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

# %%
outputPath = Path("D:/R/SeihanAnalysis/Output/Seasonality")

dayForecast99ki70.to_csv(outputPath / "dayForecast99ki70.csv")



