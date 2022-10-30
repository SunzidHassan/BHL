# %% [markdown]
# If you get stuck, ask in stackoverflow!

# %%
import numpy as np
from scipy.optimize import minimize

# %%
#Sample optimization

def objective_fun(x):
    x1 = x[0]
    x2 = x[1]
    return x1**2 + x1*x2

def equality_constraint(x):
    x1=x[0]
    x2=x[1]
    return x1**3 + x1*x2 - 100

def inequality_constraint(x):
    x1=x[0]
    x2=x[1]
    return x1**2 + x2 - 50

bounds_x1 = (-100, 100)
bounds_x2 = (-100, 100)

bounds = [bounds_x1, bounds_x2]

constraint1 = {'type': 'eq', 'fun': equality_constraint}
constraint2 = {'type': 'ineq', 'fun': inequality_constraint}

constraint = [constraint1, constraint2]

x0 = [1, 1]

result = minimize(objective_fun, x0, method='SLSQP', bounds=bounds, constraints=constraint)

print(result)


# %%
# Short version

objective_fnc = lambda x: x[0]**2 + x[0]*x[1]

constraint = [{'type': 'eq', 'fun': lambda x: x[0]**3 + x[0]*x[1] - 100},
            {'type': 'ineq', 'fun': lambda x: x[0]**2 + x[1] - 50}]

bounds = [(-100, 100), (-100, 100)]

x0 = [1, 1]

result = minimize(objective_fun, x0, method='SLSQP', bounds=bounds, constraints=constraint)

print(result)

# %%

from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt
import math as mt

import openpyxl

import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.signal import periodogram
#from statsmodels.graphics.tsaplots import plot_pacf

from scipy.optimize import minimize
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import StandardScaler
#scale = StandardScaler()
#from scipy import stats

from warnings import simplefilter
simplefilter("ignore")  # ignore warnings to clean up output cells

# %% [markdown]
# Reading constraint variables

# %%
data_dir = Path("D:/R/SeihanAnalysis/Input/20220524_ValueChainFlow/Optimizer")

Sales_Target = pd.read_csv(data_dir / 'Sales_Target.csv',
                           parse_dates=['Month'],
                           index_col=0)

Production_Capacity = pd.read_csv(data_dir / 'Production_Capacity.csv',
                           parse_dates=['Month'],
                           index_col=0)

# CKD_Standard_Days_Ceiling = pd.read_csv(data_dir / 'CKD_Standard_Days_Ceiling.csv',
#                            parse_dates=['Month'],
#                            index_col=0)

# CBU_Standard_Days_Ceiling = pd.read_csv(data_dir / 'CBU_Standard_Days_Ceiling.csv',
#                            parse_dates=['Month'],
#                            index_col=0)

# %%
Sales_Target.head()

# %% [markdown]
# Variables to populate

# %%
# Create empty dataframe for filling

CBU_Stock = pd.DataFrame({'Livo_Disk_CBS-Red':[0, 0, 0],
              'Month':['22-Aug', '22-Sep', '22-Oct']}).set_index('Month')

Production_Quantity = pd.DataFrame({'Livo_Disk_CBS-Red':[0, 0, 0],
              'Month':['22-Aug', '22-Sep', '22-Oct']}).set_index('Month')

CKD_Stock = pd.DataFrame({'Livo_Disk_CBS-Red':[0, 0, 0],
              'Month':['22-Aug', '22-Sep', '22-Oct']}).set_index('Month')

BHLIn_Quantity = pd.DataFrame({'Livo_Disk_CBS-Red':[0, 0, 0],
              'Month':['22-Aug', '22-Sep', '22-Oct']}).set_index('Month')

Order_Quantity = pd.DataFrame({'Livo_Disk_CBS-Red':[0, 0, 0],
              'Month':['22-Aug', '22-Sep', '22-Oct']}).set_index('Month')


# %%
def objective_fun(CBU_Stock):
    return CBU_Stock['Livo_Disk_CBS-Red'].sum()

bounds_CBU_days = (5, 15)
bounds_CKD_days = (15, 30)

bounds = [bounds_CBU_days, bounds_CKD_days]


def Order_BHLIn_eq(Order_Quantity, BHLIn_Quantity):
    for i in range(12):
        Order_Quantity[i]*0.6+Order_Quantity[i+1]*0.4-BHLIn_Quantity[i+3]
    return Order_Quantity, BHLIn_Quantity


def BHLIn_Production_CKDStock_eq(BHLIn_Quantity, Production_Quantity, CKD_Stock):
    for i in range(12):
        CKD_Stock[i] + BHLIn_Quantity[i+1] - Production_Quantity[i+1] - CKD_Stock[i+2]
    return CKD_Stock, BHLIn_Quantity, Production_Quantity


def Production_CBU_Sales_eq(Production_Quantity, CBU_Stock, Sales_Target):
    for i in range(12):
        CBU_Stock[i] + Production_Quantity[i+1] - Sales_Target[i+1] - CBU_Stock[i+1]
    return Production_Quantity, CBU_Stock


def CKD_Stock_days_eq():
    for i in range(12):
        CKD


def CBU_Stock_days_eq():


def Production_CKD_ineq():


def Production_CBU_Sales_ineq():


def Production_Lot_ineq():




###

def inequality_constraint(x):
    x1=x[0]
    x2=x[1]
    return x1**2 + x2 - 50

constraint1 = {'type': 'eq', 'fun': equality_constraint}
constraint2 = {'type': 'ineq', 'fun': inequality_constraint}

constraint = [constraint1, constraint2]

x0 = [1, 1]

result = minimize(objective_fun, x0, method='SLSQP', bounds=bounds, constraints=constraint)

print(result)

# %% [markdown]
# ## Simple optimization

# %% [markdown]
# Given: s1 = 950, s2 = 650, s3 = 750
# Solve just CBU problem:
# 
# 
# Equality constrinst
# starting inventory (i0) + production (p1) = sales (s1) + ending inventory (i1)
# starting inventory (i1) + production (p2) = sales (s2) + ending inventory (i2)
# starting inventory (i2) + production (p3) = sales (s3) + ending inventory (i3)
# 
# 
# Bound constraint:
# Starting inventory, i0 = 300, i3 = 200
# (s2/30 * 5) >= i1 >= (s2/30 * 15)
# (s3/30 * 5) >= i1 >= (s3/30 * 15)
# 
# 

# %%
objective_fun = lambda x: x[0]

constraint = [{'type': 'eq', 'fun': lambda x: (x[0]/1) - (x[0]//1)},
                {'type': 'ineq', 'fun': lambda x: (x*10) - 75}]

bounds = [(5, 10)]

x = [7.5]
#x = [8.5]

result = minimize(objective_fun,
                    x,
                    bounds=bounds,
                    constraints=constraint)

print(result)

# %% [markdown]
# One run

# %%
#First run
demand = [4.6, 4.9]
CBUInv = [1.3]
x = [1]

objective_fun = lambda x: x[0]
constraint = [{'type': 'ineq', 'fun': lambda x: x[0] - (demand[0] + CBUInv[0] - demand[1]*.2)}]
bounds = [(0, 10)]
result = minimize(objective_fun,
                    x,
                    bounds=bounds,
                    constraints=constraint)

CBUInv.append(mt.ceil(result.fun) + CBUInv[0] - demand[0])
x[0] = mt.ceil(result.fun)
print(CBUInv, x)

# %% [markdown]
# Multiple run

# %%
demand = [4.6, 4.9, 5.3, 3.3]
CBUInv = [1.3]
inv_req = [0, .2, .3, .1]
prod = [0] * 3
bounds = [(0, 10),
        (0, 10),
        (0, 10)]

for i in range(3):
    objective_fun = lambda x: x[i]
    constraint = [{'type': 'ineq', 'fun': lambda prod: (prod[i] + CBUInv[i]) - (demand[i] + demand[i+1]*inv_req[i+1])}]
    
    result = minimize(objective_fun,
                    prod,
                    bounds=bounds,
                    constraints=constraint)
    
    prod[i] = mt.ceil(result.fun)
    CBUInv.append(prod[i] + CBUInv[i] - demand[i])
    
print(CBUInv, prod)

# %% [markdown]
# Actual value to lotsize fraction to actual value again

# %%
actualDemand = [460, 690, 230, 330]
startingInventory = [130, 0, 0, 0]
lotSize = 100

demand = np.divide(actualDemand, lotSize)
CBUInv = np.divide(startingInventory, lotSize)

inv_req = [0, .2, .2, .2]
prod = [0] * 3
bounds = [(0, 10),
        (0, 10),
        (0, 10)]

for i in range(len(actualDemand)-1):
    objective_fun = lambda x: x[i]
    constraint = [{'type': 'ineq', 'fun': lambda prod: (prod[i]) - (- CBUInv[i] + demand[i] + demand[i+1]*inv_req[i+1])}]
    
    result = minimize(objective_fun,
                    prod,
                    bounds=bounds,
                    constraints=constraint)
    
    prod[i] = mt.ceil(result.fun)
    CBUInv[i+1] = prod[i] + CBUInv[i] - demand[i]

inventory = np.multiply(CBUInv, lotSize)
production = np.multiply(prod, lotSize)

print(inventory, production)

# %% [markdown]
# As a function

# %%
def productionPlanner(actualDemand, startingInventory, lotSize, inv_req, bounds):
    prod = [0] * (len(actualDemand)-1)
    demand = np.divide(actualDemand, lotSize)
    CBUInv = np.divide(startingInventory, lotSize)

    
    for i in range(len(actualDemand)-1):
        objective_fun = lambda x: x[i]
        constraint = [{'type': 'ineq', 'fun': lambda prod: (prod[i]) - (- CBUInv[i] + demand[i] + demand[i+1]*inv_req[i+1])}]
    
        result = minimize(objective_fun,
                    prod,
                    bounds=bounds,
                    constraints=constraint)
    
        prod[i] = mt.ceil(result.fun)
        CBUInv[i+1] = prod[i] + CBUInv[i] - demand[i]

    inventory = np.multiply(CBUInv, lotSize)
    production = np.multiply(prod, lotSize)

    print(inventory, production)

# %%
actualDemand = [460, 690, 230, 330]
startingInventory = [130, 0, 0, 0]
lotSize = 100
bounds = [(0, 10)] * (len(actualDemand)-1)
inv_req = [.2] * len(actualDemand)

productionPlanner(actualDemand, startingInventory, lotSize, inv_req, bounds)

# %% [markdown]
# Full constrainst breakdown: sales < Inventory < Production < CKD < BHL In forecast < order

# %%
#order to CKD in
def order2ckd(order, ckd):
    ckdFrcst = [0] * (len(order)+3)
    carryover = [0] * (len(order)+3)

    for i in range(len(order)-1):
        ckdFrcst[i+3] = (0.4*order[i+1]) + (0.6*order[i])

    for i in range(len(order)+3):
        if(ckd[i] == 0):
            ckd[i] = ckdFrcst[i]
    
    for i in range(3, len(order)+3):
        if(ckd[i] != 0):
            carryover[i] = ckdFrcst[i] - ckd[i]

    return ckd, carryover

# %%
#order to ckd inventory function
order = [10, 30, 20, 40]
ckd = [10, 20, 30, 20, 0, 0, 0, 0]
order2ckd(order, ckd)

# %% [markdown]
# Production requirement to order planner

# %%
production = [10, 20, 30, 40, 50, 60]
ckd = [0, 0, 0] + production
order = [0] * (len(ckd)+3)

for i in range(len(production)):
    order[i] = (ckd[i+2]*0.4) + (ckd[i+3] * .6)

print(order, ckd)

# %% [markdown]
# Add lot size consideration in order-ckd

# %%
production = np.array([300, 400, 300, 600, 500, 200])
lotSize = 100

prodLot = np.divide(production, lotSize)

ckd = np.concatenate((np.array([0, 0, 0]), prodLot))

order = np.array([0.] * (len(ckd)+3))

for i in range(len(production)):
    order[i] = np.ceil((0.4*ckd[i+2])) +  + (ckd[i+3]-(np.ceil(0.4*ckd[i+3])))

order = np.multiply(order, lotSize)
ckd = np.multiply(ckd, lotSize)
print(order, ckd)

# %% [markdown]
# Incorporate production requirement to order requriement in sales target to production requriement function

# %%
def productionPlanner(actualDemand, CBUInventory, lotSize, inv_req, bounds):
    prod = [0.] * (len(actualDemand)-1)
    demand = np.divide(actualDemand, lotSize)
    CBUInv = np.divide(CBUInventory, lotSize)

    
    for i in range(len(actualDemand)-1):
        objective_fun = lambda x: x[i]
        constraint = [{'type': 'ineq', 'fun': lambda prod: (prod[i]) - (- CBUInv[i] + demand[i] + demand[i+1]*inv_req[i+1])}]
    
        result = minimize(objective_fun,
                    prod,
                    bounds=bounds,
                    constraints=constraint)
    
        prod[i] = mt.ceil(result.fun)
        CBUInv[i+1] = prod[i] + CBUInv[i] - demand[i]
    
    production = np.multiply(prod, lotSize)

    ckd = np.concatenate((np.array([0, 0, 0]), prod))
    order = np.array([0.] * (len(ckd)+3))
    for i in range(len(production)):
        order[i] = np.ceil((0.4*ckd[i+2])) +  + (ckd[i+3]-(np.ceil(0.4*ckd[i+3])))
    
    productionSl = np.concatenate((np.array([0, 0, 0]), production))
    order = np.multiply(order, lotSize)
    
    
    return order, production, CBUInv, actualDemand

# %%
def productionPlanner(actualDemand, CBUInventory, lotSize, inv_req, bounds):
    prod = [0.] * (len(actualDemand)-1)
    demand = np.divide(actualDemand, lotSize)
    CBUInv = np.divide(CBUInventory, lotSize)
    
    for i in range(len(actualDemand)-1):
        objective_fun = lambda x: x[i]
        constraint = [{'type': 'ineq', 'fun': lambda prod: (prod[i]) - (- CBUInv[i] + demand[i] + demand[i+1]*inv_req[i+1])}]
    
        result = minimize(objective_fun,
                    prod,
                    bounds=bounds,
                    constraints=constraint)
    
        prod[i] = mt.ceil(result.fun)
        CBUInv[i+1] = prod[i] + CBUInv[i] - demand[i]
    
    order = np.array([0.] * (len(prod)+3))
    for i in range(len(prod)):
        order[i] = np.ceil((0.4*prod[i+2])) +  + (prod[i+3]-(np.ceil(0.4*prod[i+3])))
    
    production = np.multiply(prod, lotSize)
    order = np.multiply(order, lotSize)
    
    return order, production, CBUInv, actualDemand

# %%
actualDemand = [0, 0, 0, 460, 690, 230]
CBUInventory = [0, 0, 0, 0, 0, 0]
lotSize = 100
bounds = [(0, 10)] * (len(actualDemand)-1)
inv_req = [.2] * len(actualDemand)

results = productionPlanner(actualDemand, startingInventory, lotSize, inv_req, bounds)
results
# print("Inventory: ", results.inventorySl, "\nProduction: ", results.productionSl, "\nCKD", results.ckd, "\nOrder: ", results.order)

# %%
actualDemand = [0, 0, 0, 460, 690, 230, 440, 550, 660, 770]
CBUInventory = [0] * len(actualDemand)
lotSize = 100
bounds = [(0, 10)] * (len(actualDemand)-1)
inv_req = [.2] * len(actualDemand)

prod = [0.] * (len(actualDemand)-1)
demand = np.divide(actualDemand, lotSize)
CBUInv = np.divide(CBUInventory, lotSize)
    
for i in range(len(actualDemand)-1):
    objective_fun = lambda x: x[i]
    constraint = [{'type': 'ineq', 'fun': lambda prod: (prod[i]) - (- CBUInv[i] + demand[i] + demand[i+1]*inv_req[i+1])}]
        
    result = minimize(objective_fun,
                        prod,
                        bounds=bounds,
                        constraints=constraint)
        
    prod[i] = mt.ceil(result.fun)
    CBUInv[i+1] = prod[i] + CBUInv[i] - demand[i]

order = np.array([0.] * (len(prod)+3))
production = np.multiply(prod, lotSize)
CBUInv = np.multiply(CBUInv, lotSize)

for i in range(len(prod)-3):
    order[i] = np.ceil((0.4*prod[i+2])) +  + (prod[i+3]-(np.ceil(0.4*prod[i+3])))
order = np.multiply(order, lotSize)
print(actualDemand, production, CBUInv, order)

# %%
def productionPlanner(actualDemand, CBUInventory, lotSize, inv_req, bounds):
    prod = [0.] * (len(actualDemand)-1)
    demand = np.divide(actualDemand, lotSize)
    CBUInv = np.divide(CBUInventory, lotSize)
    
    for i in range(len(actualDemand)-2):
        objective_fun = lambda x: x[i]
        constraint = [{'type': 'ineq', 'fun': lambda prod: (prod[i]) - (- CBUInv[i] + demand[i] + demand[i+1]*inv_req[i+1])}]
        
        result = minimize(objective_fun,
                        prod,
                        bounds=bounds,
                        constraints=constraint)
        
        prod[i] = mt.ceil(result.fun)
        CBUInv[i+1] = prod[i] + CBUInv[i] - demand[i]

    order = np.array([0.] * (len(prod)+3))
    production = np.multiply(prod, lotSize)
    CBUInv = np.multiply(CBUInv, lotSize)

    for i in range(len(prod)-3):
        order[i] = np.ceil((0.4*prod[i+2])) +  + (prod[i+3]-(np.ceil(0.4*prod[i+3])))
    order = np.multiply(order, lotSize)

    return actualDemand, production, CBUInv, order

# %%
def productionPlanner(actualDemand, CBUInventory, lotSize, cbu_inv_req, bounds):
    prod = [0.] * (len(actualDemand)-1)
    demand = np.divide(actualDemand, lotSize)
    CBUInv = np.divide(CBUInventory, lotSize)
    
    for i in range(len(actualDemand)-2):
        objective_fun = lambda prod: prod[i]
        constraint = [{'type': 'ineq', 'fun': lambda prod: (prod[i]) - (- CBUInv[i] + demand[i] + demand[i+1]*cbu_inv_req[i+1])}]
        
        result = minimize(objective_fun,
                        prod,
                        bounds=bounds,
                        constraints=constraint)
        
        prod[i] = mt.ceil(result.fun)
        CBUInv[i+1] = prod[i] + CBUInv[i] - demand[i]

    order = np.array([0.] * (len(ckd)+3))
    production = np.multiply(ckd, lotSize)
    CBUInv = np.multiply(CBUInv, lotSize)

    for i in range(len(ckd)-3):
        order[i] = np.ceil((0.4*ckd[i+2])) +  + (ckd[i+3]-(np.ceil(0.4*ckd[i+3])))
    
    order[1] = order[1] + 1
    order[-1] = order[-1] - 1
    order = np.multiply(order, lotSize)

    return actualDemand, production, CBUInv, order

# %%
actualDemand = [0, 0, 0, 460, 690, 230, 440, 550, 660, 770]
CBUInventory = [0] * len(actualDemand)
lotSize = 100
bounds = [(0, 10)] * (len(actualDemand)-1)
cbu_inv_req = [.2] * len(actualDemand)

results = productionPlanner(actualDemand, startingInventory, lotSize, cbu_inv_req, bounds)
print("Sales Target", actualDemand, "\nProduction", production, "\nExtra CBU Inventory", CBUInv, "\nOrder Quantity", order)

# %% [markdown]
# Read data from and write to Excel

# %%


# %% [markdown]
# For multiple color

# %%


# %% [markdown]
# For multiple model-color

# %%



