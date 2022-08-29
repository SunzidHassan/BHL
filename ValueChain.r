# Notes####

## Data sources and update routine (data source, code)####
# Daily:
# BHL IN: Add/replace latest CBU,CKD & Localization Containers Arrival Details Status report
# Production and Sales:add/replace since unsold CBU's KDP data

# Monthly:
# Order: update OrderPI file with date_month_ki_BHL_Order_CKD, date_month_ki_BHL_Order_CBU files from Nahin San 
# DI: Replace the DI report from Snigdho
# Plan data: post-Seihan final Niguri

## Match####
# Match BHL in, production, sales with niguri data
# Match CKD, CBU stock data against shared/ERP report

##Todo####

###Level 1: Plan v actual
# Fix cycle active: same status for all entries of a cycle
# How are color names assigned? Shift all model color names to KDP names? (To avoid double color issue of DI?)
# Bring all in value chain visualization, and present

# Later: consider retail sales data
# Model color exhaustion: Count of seller below min stock & short amount (min shipment * number of low dealer)

### Further levels
# Deviation impact simulation > biz decision
# Plan v actual > Process improvement
# Demand prediction: use set of algorithms derived from market research
# Financial analysis


## MC Value chain Gap System and Presentation####
# L0: Goals of business planning: maximizing market share, profit, efficiency; minimizing cost, etc.
# L1: A specific goal: Market share - 1. Fully meeting existing model-color demand 2. Increasing demand
# Market share gap = lack of MC supply + lack of marketing efforts
# MC supply gap = demand prediction gap + supply deviation from plan.
# Demand prediction gap = actual current demand - predicted demand during Seihan meeting.
# Supply deviation = CBU deviation from plan
# CBU deviation = Deviation in (last month's CBU carryover + this month's production)
# This month's production deviation = deviation in CKD Stock
# Deviation in CKD stock = deviation in (last month's CKD carryover + (this month's CKD in * OK CKD ratio))
# This month's CKD in deviation = deviations in PI/DI/ BHL in date from planned order or BHL in date
# Optimize order quantity


#Import libraries####
library(readxl)
library(data.table)
library(plyr)
library(dplyr)
library(lubridate)
library(stringr)
library(tidyr)


#Part 1: Load functions#####


##PI Load Function####
OrderPILoad <- function(path){
  PI <- data.table(readxl::read_excel(path,
                                      sheet = 'Data',
                                      range = "A1:E1000"))
  
  PI$Month <- gsub("'","-", PI$Month)
  PI <- PI[!is.na(Model), ] %>%
    mutate(BHLOrderDate = my(Month)) %>%
    select(-c("Month")) %>%
    rename(PIDI_ModelName = "Model",
           PIDI_Color = "Color",
           OrderQuantity = "Order Qty",
           PIQuantity = "PI Qty")
  
  PI$OrderQuantity <- as.integer(PI$OrderQuantity)
  PI$PIQuantity <- as.integer(PI$PIQuantity)
  
  PI$PIDI_ModelName <- toupper(str_squish(PI$PIDI_ModelName))
  PI$PIDI_Color <- toupper(str_squish(PI$PIDI_Color))
  
  return(PI)
}


##DI Load function####
DILoad <- function(path){
  DI <- data.table(readxl::read_excel(path,
                                      sheet = 'Sheet1',
                                      range = "A5:AF2000")) %>%
    rename(PlantID = "Dispatch Plant",
           commercialInvoiceNo = "PC No.",
           InvoiceAmount = "Invoice Amount",
           OrderMonth = "Order Month",
           Region = "Region",
           ReceiverCountry = "Country",
           PIDI_ModelName = "Model",
           PIDI_ColorCode = "Color",
           Units = "Units",
           Category = "Category",
           ShipLine = "Ship. Line",
           ContainerSize = "Container Size",
           ContainerNo = "Container No.",
           BLNo = "BL No.",
           HMSIToDryPort_Plan = "...16",
           HMSIToDryPort_Actual = "...17",
           DryPortToWetPort_Plan = "...18",
           DryPortToWetPort_Actual = "Container Movements Status",
           VesselSailing_Plan = "...20",
           VesselSailing_Actual = "...21",
           SeaTransit = "...22",
           ETAatDestinationPort_Plan = "...23",
           ETAatDestinationPort_Actual = "...24",
           DocInvoice = "Scan Document Movement Status",
           DocPackingList = "...26",
           DocEnFCasePartList = "...27",
           DocCOO = "...28",
           DocBL = "...29",
           DocBankToBankCourierDetails = "...30",
           PaymerntStatus_Plan = "Paymernt Status",
           PaymerntStatus_Actual = "...32")
  
  DI <- DI[!is.na(commercialInvoiceNo), ]
  
  DI$HMSIToDryPort_Plan <- as.Date(as.numeric(DI$HMSIToDryPort_Plan), origin = "1899-12-30")
  DI$HMSIToDryPort_Actual <- as.Date(as.numeric(DI$HMSIToDryPort_Actual), origin = "1899-12-30")
  DI$DryPortToWetPort_Plan <- as.Date(as.numeric(DI$DryPortToWetPort_Plan), origin = "1899-12-30")
  DI$DryPortToWetPort_Actual <- as.Date(as.numeric(DI$DryPortToWetPort_Actual), origin = "1899-12-30")
  DI$VesselSailing_Plan <- as.Date(as.numeric(DI$VesselSailing_Plan), origin = "1899-12-30")
  DI$VesselSailing_Actual <- as.Date(as.numeric(DI$VesselSailing_Actual), origin = "1899-12-30")
  DI$ETAatDestinationPort_Plan <- as.Date(as.numeric(DI$ETAatDestinationPort_Plan), origin = "1899-12-30")
  DI$ETAatDestinationPort_Actual <- as.Date(as.numeric(DI$ETAatDestinationPort_Actual), origin = "1899-12-30")
  DI$DocInvoice <- as.Date(as.numeric(DI$DocInvoice), origin = "1899-12-30")
  DI$DocPackingList <- as.Date(as.numeric(DI$DocPackingList), origin = "1899-12-30")
  DI$DocEnFCasePartList <- as.Date(as.numeric(DI$DocEnFCasePartList), origin = "1899-12-30")
  DI$DocCOO <- as.Date(as.numeric(DI$DocCOO), origin = "1899-12-30")
  DI$DocBL <- as.Date(as.numeric(DI$DocBL), origin = "1899-12-30")
  DI$DocBankToBankCourierDetails <- as.Date(as.numeric(DI$DocBankToBankCourierDetails), origin = "1899-12-30")
  DI$PaymerntStatus_Plan <- as.Date(as.numeric(DI$PaymerntStatus_Plan), origin = "1899-12-30")
  DI$PaymerntStatus_Actual <- as.Date(as.numeric(DI$PaymerntStatus_Actual), origin = "1899-12-30")
  
  DI$PIDI_ColorCode <- gsub("/"," ", DI$PIDI_ColorCode)
  # DI$OrderMonth <- gsub("'","-", DI$OrderMonth)
  
  DI$PIDI_ModelName <- toupper(str_squish(DI$PIDI_ModelName))
  DI$PIDI_ColorCode <- toupper(str_squish(DI$PIDI_ColorCode))
  
  DI <- mutate(DI, DIDate = my(OrderMonth)) %>%
    mutate(BHLOrderDate = as.Date(DIDate) %m-% months(2)) %>%
    mutate(BHLOrderMonthYear = format(as.Date(BHLOrderDate), "%Y-%m"))
  
  return(DI)
}


##BHL in load####
BHLInLoad <- function(path){
  file.list <- list.files(path = path, pattern="*.xlsx", full.names = T)
  
  BHLIn <- lapply(file.list, read_excel, range = "A2:K1000") %>%
    bind_rows(.id = "id") %>%
    rename(Sl = "S.N",
           BHLIn_Model = "Model",
           BHLIn_Type = "Type",
           BHLIn_Color = "Color",
           commercialInvoiceNo = "Invoice", #substitute "-" with ""
           LotSize = "Lot Size",
           ArrivalAtCTGDate = "Arrival Date @ CTG Port",
           BHLArrivalDate = "BHL Arrival Date",
           UnloadingDate = "Unloading  Date") %>%
    filter(Sl > 0 & Sl != "TTL QTY.") %>%
    select(-c("id", 'Sl'))
  
  
  
  BHLIn$commercialInvoiceNo <- gsub("-","", BHLIn$commercialInvoiceNo)
  
  BHLIn$BHLIn_Model <- toupper(str_squish(BHLIn$BHLIn_Model))
  BHLIn$BHLIn_Type <- toupper(str_squish(BHLIn$BHLIn_Type))
  BHLIn$BHLIn_Color <- toupper(str_squish(BHLIn$BHLIn_Color))
  return(BHLIn)
}


##OK CKD Stock - MS report by production control####
# 
# CKDOKStockLoad <- function(path, range){
#   OKCKDStock <- data.table(readxl::read_excel(path,
#                                               sheet = "Stock Report",
#                                               range)) %>%
#     rename(OKCKD_Model = "Model",
#            OKCKD_Color = "Color",
#            OKCKDQuantity = "CKD Stock") %>%
#     filter(OKCKDQuantity > 0,
#            OKCKD_Color != "S.TTL")
#   OKCKDStock <- OKCKDStock[!is.na(OKCKD_Color), ] %>%
#     fill(OKCKD_Model, .direction = "down")
#   OKCKDStock$OKCKD_Model <- toupper(str_squish(OKCKDStock$OKCKD_Model))
#   OKCKDStock$OKCKD_Color <- toupper(str_squish(OKCKDStock$OKCKD_Color))
#   
#   return(OKCKDStock)
# }


##Production KDP EF report function: for both production and sales details against commercial invoice####
prodKDPdataLoad <- function(path){
  file.list <- list.files(path = path, pattern="*.xls", full.names = T)
  
  ProdKDP <- lapply(file.list, read_excel) %>%
    bind_rows(.id = "id") %>%
    rename(commercialInvoiceNo = "InvoiceNo",
           prodKDPModelCode = "ModelCode",
           prodKDPColor = "Color",
           EFJoinDate = "EFJoin",
           salesDate = "Shipment") %>%
    select(-c("SEQ", "InternalModelName", "SalesModelName", "AFOff", "VQOff", "ReturnFromBRTA")) %>%
    mutate(prodMonth = format(as.Date(EFJoinDate), "%Y-%m"),
           salesMonth = format(as.Date(salesDate), "%Y-%m"),
           count = 1)

  
  ProdKDP$EFJoinDate <- ymd(gsub(" .*","", ProdKDP$EFJoinDate))
  ProdKDP$salesDate <- ymd(gsub(" .*","", ProdKDP$salesDate))
  ProdKDP$commercialInvoiceNo <- gsub("-","", ProdKDP$commercialInvoiceNo)

  ProdKDP$prodKDPModelCode <- toupper(str_squish(ProdKDP$prodKDPModelCode))
  ProdKDP$prodKDPColor <- toupper(str_squish(ProdKDP$prodKDPColor))

  ProdKDP <- ProdKDP[!is.na(ProdKDP$VinNo), ]
  ProdKDP <- ProdKDP[!duplicated(paste(ProdKDP$VinNo, ProdKDP$EngNo, sep = "")), ]
  
  return(ProdKDP)
}




## ERP Production data with frame engine number function####

# erpProductionLoad <- function(path, date){
#   erpProduction <- data.table(readxl::read_excel(path,
#                                                  range = "A3:L90000")) %>%
#     rename(erpProduction_Model = "DA Item Name",
#            erpProductionColor = "...3",
#            erpProductionframeNo = "Frame No",
#            erpProductionengineNo = "Engine No") %>%
#     mutate(erpProductionDate = my(date),
#            erpProductionframeEngineNo = paste("ENGINE", erpProductionengineNo, "FRAME", erpProductionframeNo, sep = "")) %>%
#     select(c("erpProduction_Model", "erpProductionColor",
#              "erpProductionframeNo", "erpProductionengineNo", "erpProductionDate", "erpProductionframeEngineNo"))
#   
#   erpProduction <- erpProduction[!is.na(erpProductionengineNo), ]
#   
#   return(erpProduction)
# }


## Production team's plan data load function####

# production_plan_load <- function(path, sheet, range, monthYear){
#   prod <- data.table(readxl::read_excel(path,
#                                         sheet = sheet,
#                                         range = range)) %>%
#     rename(Production_Plan_Model_Name = "...2",
#            Production_Plan_Color = "...3",
#            Production_Plan_Quantity = "...35") %>%
#     select(c("Production_Plan_Model_Name", "Production_Plan_Color", "Production_Plan_Quantity")) %>%
#     fill(Production_Plan_Model_Name, .direction = "down") %>%
#     filter(Production_Plan_Color != "TTL",
#            Production_Plan_Quantity > 0) %>%
#     mutate(ProductionPlanDate = my(monthYear))
#   
#   prod <- prod[!is.na(Production_Plan_Color), ]
#   
#   prod$Production_Plan_Model_Name <- toupper(str_squish(prod$Production_Plan_Model_Name))
#   prod$Production_Plan_Color <- toupper(str_squish(prod$Production_Plan_Color))
#   
#   return(prod)
# }





## ERP OK CBU stock data function####

OKCBULoad <- function(summaryPath, summaryRange, detailsPath, monthYear){
  
  ERPCBUSummary <- data.table(readxl::read_excel(summaryPath,
                                                 range = summaryRange)) %>%
    select(c("Part No.", "Quantity")) %>%
    rename(CBUStock_ModelColor = "Part No.",
           CBU_Quantity = "Quantity") %>%
    dplyr::filter(CBU_Quantity > 0) %>%
    mutate(OKCBUStock_Date = my(monthYear))
  
  ERPCBUSummary$CBUStock_ModelColor <- toupper(str_squish(ERPCBUSummary$CBUStock_ModelColor))
  
  #
  # #Input CBU frame engine report
  # ERPCBUDetails <- data.table(readxl::read_excel(detailsPath,
  #                                                range = "A5:M10000")) %>%
  #   select(c("Date", "Model Name & Color")) %>%
  #   rename(CBUStock_ModelColor = "Model Name & Color",
  #          OKCBUStock_Date = "Date") %>%
  #   filter(CBUStock_ModelColor != "Model Name & Color")
  #
  # ERPCBUDetails <- ERPCBUDetails[!is.na(CBUStock_ModelColor), ]
  # ERPCBUDetails$CBUStock_ModelColor <- toupper(str_squish(ERPCBUDetails$CBUStock_ModelColor))
  #
  # OKCBUStock <- merge(ERPCBUSummary, ERPCBUDetails,
  #                       all.x = T, all.y = F)
  # OKCBUStock$OKCBUStock_Date <- as.Date(as.numeric(OKCBUStock$OKCBUStock_Date), origin = "1899-12-30")
  
  OKCBUStock <- ERPCBUSummary
  
  return(OKCBUStock)
}


##Load ERP sales data with commercial invoice, dealer and customer details####
erpSalesLoad <- function(path, sheet, date){
  erpSales <- data.table(readxl::read_excel(path,
                                            sheet = sheet,
                                            range = "A5:AI65536")) %>%
    rename(erpSalesFrameNo = "Frame No",
           erpSalesEngineNo = "Engine No",
           CustomerNo = "Customer NO",
           CustomerName = "Customer Name",
           CustomerAddress = "...6",
           erpSalesModel = "Model Name",
           erpSalesColor = "Color",
           erpSalesCommercialInvoice = "Commercial Invoice No",
           erpSalesPrice = "Price",
           erpSalesOrderNo = "Order No",
           erpSalesOrderDate = "Order      Date",
           erpSalesInvoice = "Invoice  NO",
           erpSalesInvoiceDate = "Invoice Date",
           erpSalesDeliveryOrderNo = "Delivery Order  NO",
           erpSalesDODate = "DO   Date",
           erpSalesDeliveryChallanNo = "Delivery Challan No",
           erpSalesDOChallanDate = "D.O Challan Date",
           erpSalesDOPrintCount = "D.O Print Count",
           erpSalesGatePassNo = "Gatepass  NO",
           erpSalesGatePassDate = "Gatepass  Date",
           erpSalesMushokNo = "Mushok No",
           erpSalesMushokDate ="Mushok Date",
           erpSalesSupplierCode = "Supplier Code",
           erpSalesSupplierName = "Supplier Name",
           erpSalesPONo = "PO NO.",
           erpSalesLCTrnNo = "LC Trn NO.",
           erpSalesGRNNo = "GRN NO.",
           erpSalesMONo = "MO NO.") %>%
    mutate(erpSalesDate = my(date),
           erpSalesframeEngineNo = paste("ENGINE", erpSalesEngineNo, "FRAME", erpSalesFrameNo, sep = "")) %>%
    select(c("erpSalesFrameNo", "erpSalesEngineNo", "erpSalesframeEngineNo",
             "erpSalesModel", "erpSalesColor",
             "erpSalesCommercialInvoice", "erpSalesPrice", "erpSalesDate"))

  erpSales <- erpSales[!is.na(erpSalesFrameNo), ]

  return(erpSales)
}



##Niguri load: for plan vs actual comparison, for future values####

niguriLoad <- function(path, range, budgetName, operational){
  niguri <- data.table(readxl::read_excel(path,
                                          sheet = '99Ki Niguri', range = range)) %>% #change file name if updated
    select(c("Model Name",	"Model Code",	"Genpo", "Color",	"Budget",	"Operational",	"Model Status", "Lot Size",
             "Mar-22",	"Apr-22",	"May-22",	"Jun-22", "Jul-22",
             "Aug-22",	"Sep-22",	"Oct-22",	"Nov-22",	"Dec-22",	"Jan-23",	"Feb-23",	"Mar-23")) %>%
    melt.data.table(id.vars = c("Model Name",	"Model Code",	"Genpo", "Color",	"Budget",	"Operational",	"Model Status", "Lot Size"),
                    variable.name = "MonthYear",
                    value.name = "NiguriQuantity") %>% #Unpivot Niguri
    rename(NiguriModelName = "Model Name", NiguriColor = "Color", ModelCode = "Model Code",	ModelStatus = "Model Status", Lot = "Lot Size") %>%
    mutate(count = 1, niguriDate = my(MonthYear)) %>% #Convert months to date
    filter(Budget != "LY" & Budget != "LY %"
           & ModelStatus == "Continue"
           & NiguriColor != "TTL"
           & NiguriModelName != "CKD & CBU" & NiguriModelName != "OLD Model"
           & NiguriQuantity > 0
           & Budget == budgetName
           & Operational == operational) %>% # Filter out derivative rows
    select(-c("MonthYear", "ModelStatus", "ModelCode", "Genpo", "count", "Budget", "Operational"))
  
  niguri$NiguriModelName <- toupper(str_squish(niguri$NiguriModelName))
  niguri$NiguriColor <- toupper(str_squish(niguri$NiguriColor))
  
  # niguri <- merge(niguri, internalLU, all.x = T, all.y = F,
  #                 by = c("NiguriModelName", "NiguriColor"))
  
  return(niguri)
}


#Part 2: Look up load####
## Lookup load: Not a function, but others will directly select from this data.table####
lookupload <- data.table(readxl::read_excel("Input/BHLookup.xlsx",
                                            sheet = "Combined")) %>%
  rename(PIDI_ModelName = "DI_Model_Name",
         PIDI_ColorCode = "DI_Model_Color",
         PIDI_Color = "PI_Color_Name",
         NiguriModelName = "Niguri_Model_Name",
         NiguriColor = "Niguri_Color",
         ModelCode = "ProductKDP_ModelCode")



lookupload$MS_Stock_Model <- toupper(str_squish(lookupload$MS_Stock_Model))
lookupload$MS_Stock_Color <- toupper(str_squish(lookupload$MS_Stock_Color))
lookupload$ModelCode <- toupper(str_squish(lookupload$ModelCode))
lookupload$ProductKDP_Color <- toupper(str_squish(lookupload$ProductKDP_Color))
lookupload$PIDI_ModelName <- toupper(str_squish(lookupload$PIDI_ModelName))
lookupload$PIDI_ColorCode <- toupper(str_squish(lookupload$PIDI_ColorCode))
lookupload$PIDI_Color <- toupper(str_squish(lookupload$PIDI_Color))
lookupload$Production_Plan_Color <- toupper(str_squish(lookupload$Production_Plan_Color))
lookupload$Production_Plan_Model_Name <- toupper(str_squish(lookupload$Production_Plan_Model_Name))
lookupload$OKCKD_Stock_Model_Name <- toupper(str_squish(lookupload$OKCKD_Stock_Model_Name))
lookupload$OKCKD_Stock_Color <- toupper(str_squish(lookupload$OKCKD_Stock_Color))
lookupload$Sales_Niguri_Model_Name <- toupper(str_squish(lookupload$Sales_Niguri_Model_Name))
lookupload$Sales_Niguri_Color <- toupper(str_squish(lookupload$Sales_Niguri_Color))
lookupload$ERP_Notification_Model_Name <- toupper(str_squish(lookupload$ERP_Notification_Model_Name))
lookupload$NiguriModelName <- toupper(str_squish(lookupload$NiguriModelName))
lookupload$NiguriColor <- toupper(str_squish(lookupload$NiguriColor))



##DI to PI Look up####
DIPILU <- select(lookupload, c("Sl", "PIDI_ModelName", "PIDI_ColorCode", "PIDI_Color"))
DIPILU <- DIPILU[!is.na(PIDI_ModelName), ]


## Serial look up against pidi color code####
PIColorsl <- DIPILU[!is.na(Sl), ] %>%
  select(-c("PIDI_ColorCode"))




## Look up for OK CKD stock, Production, OK CBU, Sales####
# internalLU <- select(lookupload, c("Production_Plan_Model_Name", "Production_Plan_Color",
#                                    #"OKCKD_Stock_Model_Name", "OKCKD_Stock_Color",
#                                    #"Sales_Niguri_Model_Name", "Sales_Niguri_Color",
#                                    "NiguriModelName", "NiguriColor",
#                                    "ERP_Notification_Model_Name",
#                                    "PIDI_ModelName", "PIDI_Color")) %>%
#   filter(!(is.na(Production_Plan_Model_Name) &
#              is.na(ERP_Notification_Model_Name) &
#              #is.na(OKCKD_Stock_Model_Name) &
#              #is.na(Sales_Niguri_Model_Name &
#              is.na(NiguriModelName) &
#              is.na(PIDI_ModelName)))
# 


##BHL IN Look up####
# BHLInLU <- data.table(readxl::read_excel("Input/BHLookup.xlsx",
#                                          sheet = 'Combined',
#                                          range = "A1:T100")) %>%
#   select(c("Sl", "DI_Model_Name", "DI_Model_Color", "PI_Color_Name",
#            "BHLIn_Model", "BHLIn_Type", "BHLIn_Color")) %>%
#   rename(Model = "DI_Model_Name",
#          PIDI_ColorCode = "DI_Model_Color",
#          PIDI_Color = "PI_Color_Name")
# BHLInLU <- BHLInLU[!is.na(BHLIn_Model), ]
# 
# BHLInLU$Model <- toupper(str_squish(BHLInLU$Model))
# BHLInLU$PIDI_ColorCode <- toupper(str_squish(BHLInLU$PIDI_ColorCode))
# BHLInLU$PIDI_Color <- toupper(str_squish(BHLInLU$PIDI_Color))
# 
# BHLInLU$BHLIn_Model <- toupper(str_squish(BHLInLU$BHLIn_Model))
# BHLInLU$BHLIn_Type <- toupper(str_squish(BHLInLU$BHLIn_Type))
# BHLInLU$BHLIn_Color <- toupper(str_squish(BHLInLU$BHLIn_Color))


#Part 3.1: Actual Data load####
##Load Order-PI####
orderPi <- OrderPILoad("Input/BHL_Order_PI/BHL_Order_PI.xlsx") %>%
  mutate(BHLOrderMonthYear = format(as.Date(BHLOrderDate), "%Y-%m")) %>%
  select(-c("BHLOrderDate")) %>%
  filter(OrderQuantity > 0)


##Load DI####
di <- rbind(DILoad("Input/DI/Dispatch Information Report 2021.xlsx") %>%
              select(c("commercialInvoiceNo", "OrderMonth", "PIDI_ModelName",
                       "PIDI_ColorCode", "Units", "DIDate", "BHLOrderMonthYear", "ETAatDestinationPort_Plan", "ETAatDestinationPort_Actual")),
            DILoad("Input/DI/Dispatch Information Report 2022.xlsx") %>%
              select(c("commercialInvoiceNo", "OrderMonth", "PIDI_ModelName",
                       "PIDI_ColorCode", "Units", "DIDate", "BHLOrderMonthYear", "ETAatDestinationPort_Plan", "ETAatDestinationPort_Actual")))

##Load BHL In####
BHLIn <- BHLInLoad("Input/BHLIn/") %>%
  select(c("commercialInvoiceNo", "LotSize", "BHLArrivalDate"))


##Load OK CKD Stock####
# OKCKDStock <- CKDOKStockLoad("Input/CKDData/202206/MS Stock Report Summary on 15-June-22 (Closing Stock).xlsx",
#                              "A2:C100")


##Load production KDP data for production and sales data against commercial invoice####
prodKDP <- prodKDPdataLoad("Input/ProductionKDP")


##Load ERP Production####
# erpProduction <- rbind(erpProductionLoad("Input/20220524_ValueChainFlow/erpProduction/202204moserdatemo.xls",
#                                          "02-2022"))


##Load production plan data####
# prodPlan <- rbind(production_plan_load("Input/ProductionPlan/20220605 Jun'22  Jul'22 Update Production Plan.xlsx",
#                                        "20220602 Jun'22 Proposal",
#                                        "A6:AI100",
#                                        "06-2022"),
#                   production_plan_load("Input/ProductionPlan/20220605 Jun'22  Jul'22 Update Production Plan.xlsx",
#                                        "20220604 Jul'22 Proposal",
#                                        "A6:AI100",
#                                        "07-2022"))


## Load ERP CBU OK Stock - load once for each month####
# OKCBUStock <- rbind(OKCBULoad("Input/CBUData/202206/1stSeihan/20220616_imspstk.xls",
#                               "A6:M500",
#                               "Input/CBUData/202206/1stSeihan/20220606imfrenstk.xls",
#                               "06-2022"))


##Load ERP Sales data####
# erpSales <- rbind(erpSalesLoad("Input/ERPData/erpSales/202204-20220613_sls_smry.xls", "Sheet - 1", "04-2022"),
#                   erpSalesLoad("Input/ERPData/erpSales/202204-20220613_sls_smry.xls", "Sheet - 2", "04-2022"))


#Part 3.2: Planned data load####

## Niguri production plan - for the monthly view comparison####

# remove niguri date from all, and add di date.
# Production di + 3
# Sales 

niguriPath = "Input/Niguri/202207/1stSeihan/04072022 99KI Operational Niguri_Updated.xlsx"
niguriRange = "B10:AK4000"


#niguri BHL in plan


#Niguri production plan
prodPlan <- niguriLoad(niguriPath, niguriRange,
                       "ACT+FCT", "Production") %>%
  select(-c("Lot")) %>%
  rename(Production_Plan_Quantity = "NiguriQuantity")


##Niguri Sales plan####

salesPlan <- niguriLoad(niguriPath, niguriRange,
                        "ACT+FCT", "TTL Wholesale") %>%
  select(-c("Lot")) %>%
  rename(salesQuantity = "NiguriQuantity")


##Load OK CKD Stock Niguri plan####

# okCKDStockPlan <- niguriLoad(niguriPath, niguriRange,
#                              "ACT+FCT", "CKD OK Stock") %>%
#   rename(OKCKDStockQuantity = "NiguriQuantity")


##Load OK CBU stock from Niguri Niguri plan####

# OKCBUStockPlan <- niguriLoad(niguriPath, niguriRange,
#                              "ACT+FCT", "CBU OK Stock") %>%
#   rename(CBU_Quantity = "NiguriQuantity")



## Join everything with against DI


rm(niguriPath, niguriRange)

#Part 4: Combine data ####

#Look up model color against DI color code
diMC <- merge(di, DIPILU, all.x = T, all.y = F,
              by = c("PIDI_ModelName", "PIDI_ColorCode")) %>%
  select("commercialInvoiceNo", "BHLOrderMonthYear", "DIDate",
         "Units", "PIDI_ModelName", "PIDI_Color", "ETAatDestinationPort_Plan", "ETAatDestinationPort_Actual")

diMC <- mutate(diMC, DIMonthYear = format(as.Date(DIDate), "%Y-%m"),
               transitDays = ymd(today()) - ym(DIMonthYear),
               arrivalPlan = ymd(ETAatDestinationPort_Plan)+14) %>%
  mutate(arrivalPlanDays = ymd(arrivalPlan) - ymd(today())) %>%
  select(-c("DIDate"))

diMC[!is.na(ETAatDestinationPort_Actual), c('arrivalPlan')] <- NA
diMC[!is.na(ETAatDestinationPort_Actual), c('arrivalPlanDays')] <- NA



##DI and monthly BHL IN

#DI BHL In merge
DIBHLIn <- merge(diMC, BHLIn,
                 all.x = T, all.y = T,
                 by = c("commercialInvoiceNo")) %>%
  mutate(BHLArrivalMonth = format(as.Date(BHLArrivalDate), "%Y-%m"),
         BHLArrivalDays = ymd(BHLArrivalDate) - ym(DIMonthYear),
         unusedCKDDays = ymd(today()) - ymd(BHLArrivalDate))


#Monthly DI
monthlyDI <- diMC[, .(DIQuantity = sum(Units, na.rm = T),
                      transitDays = round(mean(transitDays, na.rm = T), digits = 2),
                      arrivalPlanDays = round(mean(arrivalPlanDays, na.rm = T), digits = 2)),
                  by = .(BHLOrderMonthYear, DIMonthYear, PIDI_ModelName, PIDI_Color)]


#monthly values - order, pi is month-mode-color wise. Making DI and BHL in similar.
monthlyBHLIn <- DIBHLIn[, .(BHLInQuantity = sum(LotSize, na.rm = T),
                            BHLArrivalDays = round(mean(BHLArrivalDays, na.rm = T), digits = 2),
                            unusedCKDDays = round(mean(unusedCKDDays, na.rm = T), digits = 2)),
                        by = .(DIMonthYear, BHLArrivalMonth, PIDI_ModelName, PIDI_Color)] %>%
  filter(BHLInQuantity > 0) %>%
  dplyr::arrange(DIMonthYear, BHLArrivalMonth, PIDI_ModelName, PIDI_Color)

#Adding BHL in serial
monthlyBHLIn$monthlyBHLInSl <- 1:nrow(monthlyBHLIn)

monthlyBHLIn <- monthlyBHLIn[!is.na(DIMonthYear), ]

## Merge KDP Production with DI, and monthly KDP Production

## combine production-sales data with DI - add DI date against commercial invoice####

modelCode <- select(lookupload, c("ModelCode", "ProductKDP_Color", "PIDI_ModelName", "PIDI_Color")) %>%
  mutate(modelColor = paste(ModelCode, ProductKDP_Color, sep = ""))
modelCode <- modelCode[!is.na(ModelCode), ]
modelCode <- modelCode[!duplicated(modelColor), ]


prodKDPDI <- merge(merge(select(prodKDP, -c("VinNo", "EngNo")),
                         select(DIBHLIn, c("commercialInvoiceNo", "DIMonthYear", "BHLArrivalMonth", "BHLArrivalDate")),
                         all.x = T, all.y = F,
                         by = c("commercialInvoiceNo")),
                   modelCode,
                   all.x = T, all.y = F,
                   by.x = c("prodKDPModelCode", "prodKDPColor"),
                   by.y = c("ModelCode", "ProductKDP_Color")) %>%
  mutate(prodDateDiff = ymd(EFJoinDate) - ymd(BHLArrivalDate),
         unusedCBUDays = ymd(today()) - ymd(EFJoinDate),
         salesDateDiff = salesDate - EFJoinDate)

# remove data without DI prior to 2021
prodKDPDI <- prodKDPDI[!is.na(prodKDPDI$PIDI_ModelName), ]
prodKDPDI <- prodKDPDI[!is.na(prodKDPDI$DIMonthYear), ]
prodKDPDI <- data.table(prodKDPDI)

#get monthly production and sales
monthlyKDPProd <- prodKDPDI[!is.na(prodMonth),
                            .(monthlyKDPProd = sum(count, na.rm = T),
                              avgProdDays = round(mean(prodDateDiff, na.rm = T), digits = 2),
                              unusedCBUDays = round(mean(unusedCBUDays, na.rm = T), digits = 2)),
                            by = .(DIMonthYear, BHLArrivalMonth, prodMonth, PIDI_ModelName, PIDI_Color)]

#Arrange data according to proper model-color serial
monthlyKDPProd <- merge(monthlyKDPProd, PIColorsl,
                        all.x = T, all.y = F,
                        by = c("PIDI_ModelName", "PIDI_Color")) %>%
  dplyr::arrange(PIDI_ModelName, PIDI_Color, DIMonthYear, BHLArrivalMonth, prodMonth, Sl) %>%
  select(-c("Sl"))

#Add production serial
monthlyKDPProd$monthlyProdSl <- 1:nrow(monthlyKDPProd)

#Get monthly sales data
monthlyKDPSales <- prodKDPDI[!is.na(salesMonth), .(monthlyKDPSales = sum(count, na.rm = T),
                                                   avgSalesDays = round(mean(salesDateDiff, na.rm = T), digits = 2)),
                             by = .(DIMonthYear, BHLArrivalMonth, prodMonth, salesMonth, PIDI_ModelName, PIDI_Color)] %>%
  dplyr::arrange(PIDI_ModelName, PIDI_Color, DIMonthYear, BHLArrivalMonth, prodMonth, salesMonth)

monthlyKDPSales$monthlySalesSl <- 1:nrow(monthlyKDPSales)

#Merge Order PI with monthly DI - complete external chain

OrderPiDi <- merge(merge(orderPi, monthlyDI,
                         all.x = T,
                         all.y = T,
                         by = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear")),
                   PIColorsl, all.x = T, all.y = F,
                   by = c("PIDI_ModelName", "PIDI_Color")) %>%
  dplyr::arrange(BHLOrderMonthYear, Sl)

OrderPiDi$DiSl <- 1:nrow(OrderPiDi)


## Add monthly BHL in, KDP production and KDP sales against monthly DI and MC - complete value chain####

OrderPiDiBHLInProdSales <- merge(merge(merge(OrderPiDi, monthlyBHLIn,
                                             all.x = T, all.y = T,
                                             by = c("DIMonthYear", "PIDI_ModelName", "PIDI_Color")),
                                       monthlyKDPProd,
                                       all.x = T, all.y = T,
                                       by = c("DIMonthYear", "BHLArrivalMonth", "PIDI_ModelName", "PIDI_Color")),
                                 monthlyKDPSales,
                                 all.x = T, all.y = T,
                                 by = c("DIMonthYear", "BHLArrivalMonth", "prodMonth", "PIDI_ModelName", "PIDI_Color")) %>%
  filter(BHLOrderMonthYear > '2021-01') %>%
  filter(PIDI_ModelName != "CB SHINE SP DISC" &
           PIDI_ModelName != "DIO DLX" &
           PIDI_ModelName != "DIO" &
           PIDI_ModelName != "LIVO SELF DISC" &
           PIDI_ModelName != "LIVO SELF DRUM")

#Part 5: Add derivative values####

##DI cycle BHL in####
DICycleBHLIn <- OrderPiDiBHLInProdSales[!duplicated(monthlyBHLInSl), c("BHLInQuantity", "DiSl", "monthlyBHLInSl")]

DICycleBHLIn <- DICycleBHLIn[, .(DICycleBHLIn = sum(BHLInQuantity, na.rm = T)),
                             by = .(DiSl)] %>%
  dplyr::distinct(DiSl, .keep_all = TRUE)
DICycleBHLIn <- DICycleBHLIn[!is.na(DiSl), ]


##BHL in cycle production for calculating unused CKD####
BHLInCycleProd <- OrderPiDiBHLInProdSales[!duplicated(monthlyProdSl), c("monthlyKDPProd", "monthlyBHLInSl", "monthlyProdSl")]

BHLInCycleProd <- BHLInCycleProd[, .(BHLInCycleProd = sum(monthlyKDPProd, na.rm = TRUE)),
                                 by = .(monthlyBHLInSl)] %>%
  dplyr::distinct(monthlyBHLInSl, .keep_all = TRUE)
BHLInCycleProd <- BHLInCycleProd[!is.na(monthlyBHLInSl), ]


##Product cycle sales for calculating unused CBU####
prodCycleSales <- OrderPiDiBHLInProdSales[, .(prodCycleSales = sum(monthlyKDPSales, na.rm = T)),
                                          by = .(monthlyProdSl)] %>%
  dplyr::distinct(monthlyProdSl, .keep_all = TRUE)
prodCycleSales <- prodCycleSales[!is.na(monthlyProdSl), ]


##Merging derivative values with value chain####
OrderPiDiBHLInProdSales <- merge(merge(merge(OrderPiDiBHLInProdSales,
                                             prodCycleSales,
                                             all.x = T, all.y = F,
                                             by = c("monthlyProdSl")),
                                       BHLInCycleProd,
                                       all.x = T, all.y = F,
                                       by = c("monthlyBHLInSl")),
                                 DICycleBHLIn,
                                 all.x = T, all.y = F,
                                 by = c("DiSl"))

OrderPiDiBHLInProdSales$DICycleBHLIn[is.na(OrderPiDiBHLInProdSales$DICycleBHLIn)] <- 0
OrderPiDiBHLInProdSales$prodCycleSales[is.na(OrderPiDiBHLInProdSales$prodCycleSales)] <- 0
OrderPiDiBHLInProdSales$BHLInCycleProd[is.na(OrderPiDiBHLInProdSales$BHLInCycleProd)] <- 0

OrderPiDiBHLInProdSales <- mutate(OrderPiDiBHLInProdSales,
                                  CKDinTransit = DIQuantity - DICycleBHLIn,
                                  unusedCKD = BHLInQuantity - BHLInCycleProd,
                                  unusedCBU = monthlyKDPProd - prodCycleSales,
                                  BHLInPercent = BHLInQuantity/DIQuantity)

#If one is NA, the entire deduction is NA

DICycleTransit <- OrderPiDiBHLInProdSales[!duplicated(DiSl), c("CKDinTransit", "DiSl")]

DICycleTransit <- DICycleTransit[, .(DICycleTransit = sum(CKDinTransit, na.rm = T)),
                             by = .(DiSl)] %>%
  dplyr::distinct(DiSl, .keep_all = TRUE)
DICycleTransit <- DICycleTransit[!is.na(DiSl), ]



DICycleCKDStock <- OrderPiDiBHLInProdSales[!duplicated(monthlyBHLInSl), c("unusedCKD", "DiSl", "monthlyBHLInSl")]

DICycleCKDStock <- DICycleCKDStock[, .(DICycleCKDStock = sum(unusedCKD, na.rm = T)),
                             by = .(DiSl)] %>%
  dplyr::distinct(DiSl, .keep_all = TRUE)
DICycleCKDStock <- DICycleCKDStock[!is.na(DiSl), ]


##BHL in cycle production for calculating unused CKD####
DICycleCBUStock <- OrderPiDiBHLInProdSales[!duplicated(monthlyProdSl), c("unusedCBU", "DiSl", "monthlyProdSl")]

DICycleCBUStock <- DICycleCBUStock[, .(DICycleCBUStock = sum(unusedCBU, na.rm = T)),
                                 by = .(DiSl)] %>%
  dplyr::distinct(DiSl, .keep_all = TRUE)
DICycleCBUStock <- DICycleCBUStock[!is.na(DiSl), ]


OrderPiDiBHLInProdSales <- merge(merge(merge(OrderPiDiBHLInProdSales,
                                             DICycleTransit,
                                             all.x = T, all.y = F,
                                             by = c("DiSl")),
                                       DICycleCKDStock,
                                       all.x = T, all.y = F,
                                       by = c("DiSl")),
                                 DICycleCBUStock,
                                 all.x = T, all.y = F,
                                 by = c("DiSl"))

OrderPiDiBHLInProdSales$DICycleTransit[is.na(OrderPiDiBHLInProdSales$DICycleTransit)] <- 0
OrderPiDiBHLInProdSales$DICycleCKDStock[is.na(OrderPiDiBHLInProdSales$DICycleCKDStock)] <- 0
OrderPiDiBHLInProdSales$DICycleCBUStock[is.na(OrderPiDiBHLInProdSales$DICycleCBUStock)] <- 0

OrderPiDiBHLInProdSales <- mutate(OrderPiDiBHLInProdSales,
                                  cycleActive = (DICycleTransit + DICycleCKDStock + DICycleCBUStock) > 0)
OrderPiDiBHLInProdSales$cycleActive[is.na(OrderPiDiBHLInProdSales$cycleActive)] <- TRUE
OrderPiDiBHLInProdSales$cycleActive[is.na(OrderPiDiBHLInProdSales$BHLInQuantity)] <- TRUE
OrderPiDiBHLInProdSales$cycleActive[is.na(OrderPiDiBHLInProdSales$monthlyKDPProd)] <- TRUE



for(i in 1:length(OrderPiDiBHLInProdSales$DiSl)){
  if(is.na(OrderPiDiBHLInProdSales$monthlyKDPProd[i])){
    OrderPiDiBHLInProdSales$unusedCKD[i] = OrderPiDiBHLInProdSales$BHLInQuantity[i]
  }}


OrderPiDiBHLInProdSales <- select(OrderPiDiBHLInProdSales, c("PIDI_ModelName", "PIDI_Color",
                                                             "BHLOrderMonthYear", "OrderQuantity",
                                                             "PIQuantity",
                                                             "DIMonthYear", "DIQuantity",
                                                             "BHLArrivalMonth", "BHLInQuantity", "BHLInPercent", "BHLArrivalDays",
                                                             "prodMonth", "monthlyKDPProd", "avgProdDays",
                                                             "salesMonth", "monthlyKDPSales", "avgSalesDays",
                                                             "CKDinTransit", "transitDays", "arrivalPlanDays", "unusedCKD", "unusedCKDDays", "unusedCBU", "unusedCBUDays",
                                                             "prodCycleSales", "BHLInCycleProd",
                                                             "DiSl", "monthlyBHLInSl","monthlyProdSl", "monthlySalesSl",
                                                             "DICycleTransit", "DICycleCKDStock", "DICycleCBUStock", "cycleActive"))

#Part 6: Remove duplicate cells in orderpidibhlinsales report####
OrderPiDiBHLInProdSalesNonDuplicate <- OrderPiDiBHLInProdSales
OrderPiDiBHLInProdSalesNonDuplicate$duplicateDISL <- NA
OrderPiDiBHLInProdSalesNonDuplicate$duplicateProdSL <- NA
OrderPiDiBHLInProdSalesNonDuplicate$duplicateBHLInSL <- NA

#Add column: if there are duplicate DI serials
OrderPiDiBHLInProdSalesNonDuplicate <- dplyr::arrange(OrderPiDiBHLInProdSalesNonDuplicate, DiSl)
for(i in 2:length(OrderPiDiBHLInProdSalesNonDuplicate$DiSl)){
  OrderPiDiBHLInProdSalesNonDuplicate$duplicateDISL[i] <- (OrderPiDiBHLInProdSalesNonDuplicate$DiSl[i] == OrderPiDiBHLInProdSalesNonDuplicate$DiSl[i-1])
}

#Add column: if there are duplicate BHL In  serials
OrderPiDiBHLInProdSalesNonDuplicate <- dplyr::arrange(OrderPiDiBHLInProdSalesNonDuplicate, monthlyProdSl)
for(i in 2:(length(OrderPiDiBHLInProdSalesNonDuplicate$monthlyProdSl))){
  OrderPiDiBHLInProdSalesNonDuplicate$duplicateProdSL[i] <- (OrderPiDiBHLInProdSalesNonDuplicate$monthlyProdSl[i] == OrderPiDiBHLInProdSalesNonDuplicate$monthlyProdSl[i-1])
}

#Add column: if there are duplicate production serials
OrderPiDiBHLInProdSalesNonDuplicate <- dplyr::arrange(OrderPiDiBHLInProdSalesNonDuplicate, monthlyBHLInSl)
for(i in 2:(length(OrderPiDiBHLInProdSalesNonDuplicate$monthlyBHLInSl))){
  OrderPiDiBHLInProdSalesNonDuplicate$duplicateBHLInSL[i] <- (OrderPiDiBHLInProdSalesNonDuplicate$monthlyBHLInSl[i] == OrderPiDiBHLInProdSalesNonDuplicate$monthlyBHLInSl[i-1])
}

#Delete duplicate entries
for(i in 1:length(OrderPiDiBHLInProdSalesNonDuplicate$DiSl)){
  # delete external chain entries if di is duplicate
  if(isTRUE(OrderPiDiBHLInProdSalesNonDuplicate$duplicateDISL[i])){
    OrderPiDiBHLInProdSalesNonDuplicate$BHLOrderMonthYear[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$OrderQuantity[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$PIQuantity[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$DIMonthYear[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$DIQuantity[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$CKDinTransit[i] <- NA
  }
  if(isTRUE(OrderPiDiBHLInProdSalesNonDuplicate$duplicateProdSL[i])){
    OrderPiDiBHLInProdSalesNonDuplicate$BHLArrivalMonth[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$BHLInQuantity[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$prodMonth[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$monthlyKDPProd[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$avgProdDays[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$unusedCBU[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$unusedCBUDays[i] <- NA
  }
  if(isTRUE(OrderPiDiBHLInProdSalesNonDuplicate$duplicateBHLInSL[i])){
    OrderPiDiBHLInProdSalesNonDuplicate$BHLArrivalMonth[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$BHLInQuantity[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$BHLInPercent[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$BHLArrivalDays[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$unusedCKD[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$unusedCKDDays[i] <- NA
  }
  if(is.na(OrderPiDiBHLInProdSalesNonDuplicate$unusedCKD[i]) ||
     OrderPiDiBHLInProdSalesNonDuplicate$unusedCKD[i] == 0 ){
    OrderPiDiBHLInProdSalesNonDuplicate$unusedCKDDays[i] <- NA
  }
  if(is.na(OrderPiDiBHLInProdSalesNonDuplicate$unusedCBU[i]) ||
     OrderPiDiBHLInProdSalesNonDuplicate$unusedCBU[i] == 0){
    OrderPiDiBHLInProdSalesNonDuplicate$unusedCBUDays[i] <- NA
  }
  if(is.na(OrderPiDiBHLInProdSalesNonDuplicate$CKDinTransit[i]) ||
     OrderPiDiBHLInProdSalesNonDuplicate$CKDinTransit[i] == 0){
    OrderPiDiBHLInProdSalesNonDuplicate$transitDays[i] <- NA
    OrderPiDiBHLInProdSalesNonDuplicate$arrivalPlanDays[i] <- NA
  }
}

OrderPiDiBHLInProdSalesNonDuplicate <- select(OrderPiDiBHLInProdSalesNonDuplicate,
                                              c("PIDI_ModelName", "PIDI_Color",
                                                "BHLOrderMonthYear", "OrderQuantity",
                                                "PIQuantity",
                                                "DIMonthYear", "DIQuantity",
                                                "BHLArrivalMonth", "BHLInQuantity", "BHLInPercent", "BHLArrivalDays",
                                                "prodMonth", "monthlyKDPProd", "avgProdDays",
                                                "salesMonth", "monthlyKDPSales", "avgSalesDays",
                                                "CKDinTransit", "transitDays", "arrivalPlanDays", "unusedCKD", "unusedCKDDays", "unusedCBU", "unusedCBUDays",
                                                "prodCycleSales", "BHLInCycleProd",
                                                "DiSl", "monthlyBHLInSl","monthlyProdSl", "monthlySalesSl",
                                                "cycleActive", "duplicateDISL", "duplicateProdSL", "duplicateBHLInSL")) %>%
  dplyr::arrange(DiSl, monthlyBHLInSl, monthlyProdSl, monthlySalesSl) %>%
  rename(Model = "PIDI_ModelName", Color = "PIDI_Color",
         OrderMonth = "BHLOrderMonthYear", Order = "OrderQuantity",
         DIMonth = "DIMonthYear", DI = "DIQuantity",
         BHLInMonth = "BHLArrivalMonth", BHLIn = "BHLInQuantity", BHLInDays = "BHLArrivalDays",
         ProdMonth = "prodMonth", Prod = "monthlyKDPProd", ProdDays = "avgProdDays",
         SalesMonth = "salesMonth", Sales = "monthlyKDPSales", SalesDays = "avgSalesDays",
         Transit = "CKDinTransit", TransitDays = "transitDays", ArrivalPlan = "arrivalPlanDays", CKDStock = "unusedCKD", CKDDays = "unusedCKDDays", CBUStock = "unusedCBU", CBUDays = "unusedCBUDays")


#Part 7: Add planned columns####

##Adding lot size with orderpidi####
# 
# modelCode <- select(lookupload, c("Sl", "NiguriModelName", "NiguriColor", "PIDI_ModelName", "PIDI_Color"))
# modelCode <- modelCode[!is.na(NiguriModelName), ]
# modelCode <- modelCode[!is.na(NiguriColor), ]
# modelCode <- modelCode[!is.na(PIDI_ModelName), ]
# 
# 
# niguriPath = "Input/Niguri/202207/1stSeihan/04072022 99KI Operational Niguri_Updated.xlsx"
# niguriRange = "B10:AK4000"
# 
# lot <- niguriLoad(niguriPath, niguriRange,
#                   "ACT+FCT", "CKD Order") %>%
#   select("NiguriModelName", "NiguriColor", "Lot") %>%
#   distinct(NiguriModelName, NiguriColor, .keep_all = T)
# 
# lot <- merge(lot, modelCode,
#              all.x = T, all.y = F,
#              by = c("NiguriModelName", "NiguriColor")) %>%
#   select(-c("NiguriModelName", "NiguriColor")) %>%
#   rename(Model = "PIDI_ModelName", Color = "PIDI_Color")
# 
# lot <- lot[!is.na(Model), ]
# 
# 
# OrderPiDiBHLInProdSalesNonDuplicate <- merge(OrderPiDiBHLInProdSalesNonDuplicate, lot,
#               all.x = T, all.y = F,
#               by = c("Model", "Color"))
# 
# 
# 
# OrderPiDiBHLInProdSalesNonDuplicate$IdealBHLInMonth <- NA
# OrderPiDiBHLInProdSalesNonDuplicate$IdealBHLIn <- NA
# 
# OrderPiDiBHLInProdSalesNonDuplicate$IdealProdMonth <- NA
# OrderPiDiBHLInProdSalesNonDuplicate$IdealProd <- NA
# 
# OrderPiDiBHLInProdSalesNonDuplicate$IdealSalesMonth <- NA
# OrderPiDiBHLInProdSalesNonDuplicate$IdealSales <- NA
# 
# 
# for(i in 1:length(OrderPiDiBHLInProdSalesNonDuplicate$DiSl)){
#   if(isFALSE(OrderPiDiBHLInProdSalesNonDuplicate$duplicateDISL[i])){
#     OrderPiDiBHLInProdSalesNonDuplicate$IdealBHLInMonth[i] = ym(OrderPiDiBHLInProdSalesNonDuplicate$OrderMonth[i])+months(3)
#     OrderPiDiBHLInProdSalesNonDuplicate$IdealProdMonth[i] = ym(OrderPiDiBHLInProdSalesNonDuplicate$OrderMonth[i])+months(4)
#     OrderPiDiBHLInProdSalesNonDuplicate$IdealSalesMonth[i] = ym(OrderPiDiBHLInProdSalesNonDuplicate$OrderMonth[i])+months(4)
#     
#     OrderPiDiBHLInProdSalesNonDuplicate$IdealBHLIn[i] = (ceiling((OrderPiDiBHLInProdSalesNonDuplicate$Order[i]/OrderPiDiBHLInProdSalesNonDuplicate$Lot[i]) * 0.4)) * OrderPiDiBHLInProdSalesNonDuplicate$Lot[i]
#     OrderPiDiBHLInProdSalesNonDuplicate$IdealProd[i] = (ceiling((OrderPiDiBHLInProdSalesNonDuplicate$Order[i]/OrderPiDiBHLInProdSalesNonDuplicate$Lot[i]) * 0.4)) * OrderPiDiBHLInProdSalesNonDuplicate$Lot[i]
#     OrderPiDiBHLInProdSalesNonDuplicate$IdealSales[i] = (ceiling((OrderPiDiBHLInProdSalesNonDuplicate$Order[i]/OrderPiDiBHLInProdSalesNonDuplicate$Lot[i]) * 0.4)) * OrderPiDiBHLInProdSalesNonDuplicate$Lot[i]
#     }
#   if(isTRUE(OrderPiDiBHLInProdSalesNonDuplicate$duplicateDISL[i])){
#     OrderPiDiBHLInProdSalesNonDuplicate$IdealBHLInMonth[i] = ym(OrderPiDiBHLInProdSalesNonDuplicate$OrderMonth[i])+months(4)
#     OrderPiDiBHLInProdSalesNonDuplicate$IdealProdMonth[i] = ym(OrderPiDiBHLInProdSalesNonDuplicate$OrderMonth[i])+months(5)
#     OrderPiDiBHLInProdSalesNonDuplicate$IdealSalesMonth[i] = ym(OrderPiDiBHLInProdSalesNonDuplicate$OrderMonth[i])+months(5)
#     
#     OrderPiDiBHLInProdSalesNonDuplicate$IdealBHLIn[i] = OrderPiDiBHLInProdSalesNonDuplicate$Order[i] - ((ceiling((OrderPiDiBHLInProdSalesNonDuplicate$Order[i]/OrderPiDiBHLInProdSalesNonDuplicate$Lot[i]) * 0.4)) * OrderPiDiBHLInProdSalesNonDuplicate$Lot[i])
#     OrderPiDiBHLInProdSalesNonDuplicate$IdealProdMonth[i] = OrderPiDiBHLInProdSalesNonDuplicate$Order[i] - ((ceiling((OrderPiDiBHLInProdSalesNonDuplicate$Order[i]/OrderPiDiBHLInProdSalesNonDuplicate$Lot[i]) * 0.4)) * OrderPiDiBHLInProdSalesNonDuplicate$Lot[i])
#     OrderPiDiBHLInProdSalesNonDuplicate$IdealSalesMonth[i] = OrderPiDiBHLInProdSalesNonDuplicate$Order[i] - ((ceiling((OrderPiDiBHLInProdSalesNonDuplicate$Order[i]/OrderPiDiBHLInProdSalesNonDuplicate$Lot[i]) * 0.4)) * OrderPiDiBHLInProdSalesNonDuplicate$Lot[i])
#   }}
# 
# 
# 









# if duplicate DI sl is true: 40% of order, production, sales
# if duplicate di sl is false: 60%

#Part 8: Melt data for reporting####

#Selecting order data
orderQuant <- select(OrderPiDiBHLInProdSales, c("PIDI_ModelName", "PIDI_Color", "DiSl",
                                                "BHLOrderMonthYear",
                                                "OrderQuantity")) %>%
  filter(OrderQuantity > 0) %>%
  mutate(monthYear = BHLOrderMonthYear) %>%
  distinct(PIDI_ModelName, PIDI_Color, DiSl, BHLOrderMonthYear, monthYear, OrderQuantity)

#Selecting DI data
diQuant <- select(OrderPiDiBHLInProdSales, c("PIDI_ModelName", "PIDI_Color", "DiSl",
                                             "BHLOrderMonthYear",
                                             "DIMonthYear", "DIQuantity")) %>%
  filter(DIQuantity > 0) %>%
  rename(monthYear = DIMonthYear) %>%
  distinct(PIDI_ModelName, PIDI_Color, BHLOrderMonthYear, DiSl, monthYear, DIQuantity)

#Selecting BHL in data
BHLInQuant <- select(OrderPiDiBHLInProdSales, c("PIDI_ModelName", "PIDI_Color", "DiSl", "monthlyBHLInSl",
                                                "BHLOrderMonthYear",
                                                "BHLArrivalMonth", "BHLInQuantity")) %>%
  filter(BHLInQuantity > 0) %>%
  rename(monthYear = BHLArrivalMonth) %>%
  distinct(PIDI_ModelName, PIDI_Color, BHLOrderMonthYear, DiSl, monthlyBHLInSl, monthYear, BHLInQuantity)

#Selecting transit data
CKDinTransit <- select(OrderPiDiBHLInProdSales, c("PIDI_ModelName", "PIDI_Color", "DiSl",
                                                  "BHLOrderMonthYear",
                                                  "DIMonthYear", "CKDinTransit")) %>%
  filter(CKDinTransit > 0) %>%
  rename(monthYear = DIMonthYear) %>%
  distinct(PIDI_ModelName, PIDI_Color, BHLOrderMonthYear, DiSl, monthYear, CKDinTransit)


#Selecting unused CKD data
unusedCKDQuant <- select(OrderPiDiBHLInProdSales, c("PIDI_ModelName", "PIDI_Color", "DiSl", "monthlyBHLInSl", "monthlyProdSl",
                                                    "BHLOrderMonthYear",
                                                    "BHLArrivalMonth", "unusedCKD")) %>%
  filter(unusedCKD > 0) %>%
  rename(monthYear = BHLArrivalMonth) %>%
  distinct(PIDI_ModelName, PIDI_Color, BHLOrderMonthYear, DiSl, monthlyBHLInSl, monthlyProdSl, monthYear, unusedCKD)

#Selecting production data
prodQuant <- select(OrderPiDiBHLInProdSales, c("PIDI_ModelName", "PIDI_Color", "DiSl", "monthlyBHLInSl", "monthlyProdSl",
                                               "BHLOrderMonthYear",
                                               "prodMonth", "monthlyKDPProd")) %>%
  filter(monthlyKDPProd > 0) %>%
  rename(monthYear = prodMonth) %>%
  distinct(PIDI_ModelName, PIDI_Color, BHLOrderMonthYear, DiSl, monthlyBHLInSl, monthlyProdSl, monthYear, monthlyKDPProd)

#Selecting production days data
prodDays <- select(OrderPiDiBHLInProdSales, c("PIDI_ModelName", "PIDI_Color", "DiSl", "monthlyBHLInSl", "monthlyProdSl",
                                              "BHLOrderMonthYear",
                                              "prodMonth", "avgProdDays")) %>%
  filter(avgProdDays > 0) %>%
  rename(monthYear = prodMonth) %>%
  distinct(PIDI_ModelName, PIDI_Color, BHLOrderMonthYear, DiSl, monthlyBHLInSl, monthlyProdSl, monthYear, avgProdDays)

#Selecting sales data
salesQuant <- select(OrderPiDiBHLInProdSales, c("PIDI_ModelName", "PIDI_Color", "DiSl", "monthlyBHLInSl", "monthlyProdSl",
                                                "BHLOrderMonthYear",
                                                "salesMonth", "monthlyKDPSales")) %>%
  filter(monthlyKDPSales > 0) %>%
  rename(monthYear = salesMonth) %>%
  distinct(PIDI_ModelName, PIDI_Color, BHLOrderMonthYear, DiSl, monthlyBHLInSl, monthlyProdSl, monthYear, monthlyKDPSales)

#Selecting sales days data
salesDays <- select(OrderPiDiBHLInProdSales, c("PIDI_ModelName", "PIDI_Color", "DiSl", "monthlyBHLInSl", "monthlyProdSl",
                                               "BHLOrderMonthYear",
                                               "salesMonth", "avgSalesDays")) %>%
  filter(avgSalesDays > 0) %>%
  rename(monthYear = salesMonth) %>%
  distinct(PIDI_ModelName, PIDI_Color, BHLOrderMonthYear, DiSl, monthlyBHLInSl, monthlyProdSl, monthYear, avgSalesDays)

#Selecting unused CBU data
unusedCBUQuant <- select(OrderPiDiBHLInProdSales, c("PIDI_ModelName", "PIDI_Color", "DiSl", "monthlyBHLInSl", "monthlyProdSl",
                                                    "BHLOrderMonthYear",
                                                    "prodMonth", "unusedCBU")) %>%
  filter(unusedCBU > 0) %>%
  rename(monthYear = prodMonth) %>%
  distinct(PIDI_ModelName, PIDI_Color, BHLOrderMonthYear, DiSl, monthlyBHLInSl, monthlyProdSl, monthYear, unusedCBU)



#Merging all
OrderPiDiBHLInProdSalesFlat <- merge(merge(merge(merge(merge(merge(merge(merge(merge(orderQuant, diQuant,
                                                                                     all = T,
                                                                                     by = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear", "DiSl", "monthYear")),
                                                                               CKDinTransit,
                                                                               all = T,
                                                                               by = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear", "DiSl", "monthYear")),
                                                                         BHLInQuant,
                                                                         all = T,
                                                                         by = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear", "DiSl", "monthYear")),
                                                                   prodQuant,
                                                                   all = T,
                                                                   by = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear", "DiSl", "monthYear", "monthlyBHLInSl")),
                                                             unusedCKDQuant,
                                                             all = T,
                                                             by = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear", "DiSl", "monthYear", "monthlyBHLInSl", "monthlyProdSl")),
                                                       prodDays,
                                                       all = T,
                                                       by = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear", "DiSl", "monthYear", "monthlyBHLInSl", "monthlyProdSl")),
                                                 salesQuant,
                                                 all = T,
                                                 by = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear", "DiSl", "monthYear", "monthlyBHLInSl", "monthlyProdSl")),
                                           salesDays,
                                           all = T,
                                           by = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear", "DiSl", "monthYear", "monthlyBHLInSl", "monthlyProdSl")),
                                     unusedCBUQuant,
                                     all = T,
                                     by = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear", "DiSl", "monthYear", "monthlyBHLInSl", "monthlyProdSl"))


OrderPiDiBHLInProdSalesFlat <- melt.data.table(OrderPiDiBHLInProdSalesFlat,
                                               id.vars = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear", "DiSl", "monthlyBHLInSl", "monthlyProdSl", "monthYear"),
                                               variable.name = "Type",
                                               value.name = "Quantity")

OrderPiDiBHLInProdSalesFlat <- OrderPiDiBHLInProdSalesFlat[!is.na(Quantity)]

#Remove production serial from BHL In
OrderPiDiBHLInProdSalesFlat[Type == "BHLInQuantity", c("monthlyProdSl")] <- NA

# Arrange values for adding sales serial
OrderPiDiBHLInProdSalesFlat <- OrderPiDiBHLInProdSalesFlat %>%
  dplyr::arrange(PIDI_ModelName, PIDI_Color, BHLOrderMonthYear, DiSl, monthlyBHLInSl, monthlyProdSl, monthYear)

# Add sales serial
salesFilter <- filter(OrderPiDiBHLInProdSalesFlat, !is.na(Quantity), Type == "monthlyKDPSales")
salesFilter$monthlySalesSl <- 1:nrow(salesFilter)

# Merge sales serial with flat data
OrderPiDiBHLInProdSalesFlat <- merge(OrderPiDiBHLInProdSalesFlat, salesFilter,
                                     by = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear",
                                            "DiSl", "monthlyBHLInSl", "monthlyProdSl",
                                            "monthYear", "Type", "Quantity"),
                                     all.x = T, all.y = F)

# Rank of multiple BHL In, Production, Sales, etc.####

# Input data for BHL In rank
intemp1 <- OrderPiDiBHLInProdSalesFlat %>% filter(!is.na(Quantity), Type == "BHLInQuantity")

# BHL in rank
outtemp1 <- intemp1 %>% group_by(DiSl) %>%
  mutate(monthlyBHLInSlRank = order(order(DiSl, monthlyBHLInSl, decreasing = FALSE))) %>%
  select(c("DiSl", "monthlyBHLInSl", "monthlyBHLInSlRank"))

# Input data for production rank
intemp2 <- OrderPiDiBHLInProdSalesFlat %>% filter(!is.na(Quantity), Type == "monthlyKDPProd")

# Production rank
outtemp2 <- intemp2 %>% group_by(DiSl, monthlyBHLInSl) %>%
  mutate(monthlyProdSlRank = order(order(DiSl, monthlyBHLInSl, monthlyProdSl, decreasing = FALSE))) %>%
  select(c("DiSl", "monthlyBHLInSl", "monthlyProdSl", "monthlyProdSlRank"))

# Input data for sales rank
intemp3 <- OrderPiDiBHLInProdSalesFlat %>% filter(!is.na(Quantity), Type == "monthlyKDPSales")

# Sales rank
outtemp3 <- intemp3 %>% group_by(DiSl, monthlyBHLInSl, monthlyProdSl) %>%
  mutate(monthlySalesSlRank = order(order(DiSl, monthlyBHLInSl, monthlyProdSl, monthlySalesSl, decreasing = FALSE))) %>%
  select(c("DiSl", "monthlyBHLInSl", "monthlyProdSl", "monthlySalesSl", "monthlySalesSlRank"))

# Didn't worked: merging rank together, and then adding with main data
# outtemp <- merge(merge(outtemp3, outtemp2, all = T),
#                  outtemp1, all = T)
# 
# 
# OrderPiDiBHLInProdSalesFlat2 <- merge(OrderPiDiBHLInProdSalesFlat, outtemp,
#                                       all.x = T, all.y = F,
#                                       by = c("DiSl", "monthlyBHLInSl", "monthlyProdSl", "monthlySalesSl")) %>%
#   select(c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear",
#            "DiSl", "monthlyBHLInSl", "monthlyProdSl", "monthlySalesSl",
#            "Type", "monthYear", "Quantity",
#            "monthlyBHLInSlRank", "monthlyProdSlRank", "monthlySalesSlRank"))

# Merging BHL in rank with main data
OrderPiDiBHLInProdSalesFlat2 <- merge(OrderPiDiBHLInProdSalesFlat, outtemp1,
                                      all.x = T, all.y = F,
                                      by = c("DiSl", "monthlyBHLInSl"))

# Merging production rank with main data
OrderPiDiBHLInProdSalesFlat2 <- merge(OrderPiDiBHLInProdSalesFlat2, outtemp2,
                                      all.x = T, all.y = F,
                                      by = c("DiSl", "monthlyBHLInSl", "monthlyProdSl"))

# Merging sales rank with main data, organizing columns
OrderPiDiBHLInProdSalesFlat2 <- merge(OrderPiDiBHLInProdSalesFlat2, outtemp3,
                                      all.x = T, all.y = F,
                                      by = c("DiSl", "monthlyBHLInSl", "monthlyProdSl", "monthlySalesSl")) %>%
  select(c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear",
           "DiSl", "monthlyBHLInSl", "monthlyProdSl", "monthlySalesSl",
           "Type", "monthYear", "Quantity",
           "monthlyBHLInSlRank", "monthlyProdSlRank", "monthlySalesSlRank"))

#Part 9: Export####

write.csv(OrderPiDiBHLInProdSales, "Output/valueChain/OrderPiDiBHLInProdSales.csv", na = "NA")

write.csv(OrderPiDiBHLInProdSalesFlat2, "Output/valueChain/OrderPiDiBHLInProdSalesFlat.csv", na = "NA")

write.csv(OrderPiDiBHLInProdSalesNonDuplicate, "Output/valueChain/OrderPiDiBHLInProdSalesnoDuplicate.csv", na = "NA")



# rm(diMC, i, BHLInCycleProd, BHLInQuant, DIBHLIn, DICycleBHLIn, diMC, DIPILU, diQuant,
#    modelCode, orderPi, OrderPiDi, orderQuant, PIColorsl, prodCycleSales, prodDays, prodKDPDI,
#    prodPlan, prodQuant, salesDays, salesQuant, salesPlan, unusedCBUQuant, unusedCKDQuant)


#Part 10: Monthly plan v actual####

# monthly BHL in, production, sales plan vs actual values

# Actual values - take sum over model, color, year/month

monthlyBHLIn <- OrderPiDiBHLInProdSalesNonDuplicate[,.(monthlyBHLIn = sum(BHLIn, na.rm = T)),
                                                    by = .(Model, Color, BHLInMonth)] %>%
  rename(month = BHLInMonth)
monthlyBHLIn$month <- ym(monthlyBHLIn$month)

monthlyProd <- OrderPiDiBHLInProdSalesNonDuplicate[,.(monthlyProd = sum(Prod, na.rm = T)),
                                                    by = .(Model, Color, ProdMonth)] %>%
  rename(month = ProdMonth)
monthlyProd$month <- ym(monthlyProd$month)

monthlySales <- OrderPiDiBHLInProdSalesNonDuplicate[,.(monthlySales = sum(Sales, na.rm = T)),
                                                    by = .(Model, Color, SalesMonth)] %>%
  rename(month = SalesMonth)
monthlySales$month <- ym(monthlySales$month)

modelCode <- select(lookupload, c("Sl", "NiguriModelName", "NiguriColor", "PIDI_ModelName", "PIDI_Color"))
modelCode <- modelCode[!is.na(NiguriModelName), ]
modelCode <- modelCode[!is.na(NiguriColor), ]
modelCode <- modelCode[!is.na(PIDI_ModelName), ]

# Plan: order times 40/60

niguriPath = "Input/Niguri/202207/1stSeihan/04072022 99KI Operational Niguri_Updated.xlsx"
niguriRange = "B10:AK4000"

lot <- niguriLoad(niguriPath, niguriRange,
                       "ACT+FCT", "CKD Order") %>%
  select("NiguriModelName", "NiguriColor", "Lot") %>%
  distinct(NiguriModelName, NiguriColor, .keep_all = T)

lot <- merge(lot, modelCode,
             all.x = T, all.y = F,
             by = c("NiguriModelName", "NiguriColor")) %>%
  select(-c("NiguriModelName", "NiguriColor")) %>%
  rename(Model = "PIDI_ModelName", Color = "PIDI_Color")

lot <- lot[!is.na(Model), ]


monthlyPlan <- merge(select(OrderPiDiBHLInProdSalesNonDuplicate, c("Model", "Color", "OrderMonth", "Order")),
                     lot,
                     all.x = T, all.y = F,
                     by = c("Model", "Color")) %>%
  distinct(Model, Color, OrderMonth, .keep_all = T)

monthlyPlan <- monthlyPlan[!is.na(Order), ]
monthlyPlan <- monthlyPlan[!is.na(Lot), ]


BHLInPlan <- rbind(mutate(monthlyPlan, month = ym(OrderMonth)+months(3),
                           BHLInPlan = (ceiling((Order/Lot) * 0.4)) * Lot),
                    mutate(monthlyPlan, month = ym(OrderMonth)+months(4),
                           BHLInPlan = Order - ((ceiling((Order/Lot) * 0.4)) * Lot))) %>%
  select(-c("Order", "OrderMonth", "Lot", "Sl"))

BHLInPlan <- BHLInPlan[,.(BHLInPlan = sum(BHLInPlan, na.rm = T)),
                       by = .(Model, Color, month)]


ProdPlan <- rbind(mutate(monthlyPlan, month = ym(OrderMonth)+months(4),
                          ProdPlan = (ceiling((Order/Lot) * 0.4)) * Lot),
                   mutate(monthlyPlan, month = ym(OrderMonth)+months(5),
                          ProdPlan = Order - ((ceiling((Order/Lot) * 0.4)) * Lot))) %>%
  select(-c("Order", "OrderMonth", "Lot", "Sl"))

ProdPlan <- ProdPlan[,.(ProdPlan = sum(ProdPlan, na.rm = T)),
                       by = .(Model, Color, month)]%>%
  mutate(SalesPlan = ProdPlan)
  

PlanvActual <- merge(merge(merge(merge(monthlyBHLIn, monthlyProd,
                                       all = T,
                                       by = c("Model", "Color", "month")),
                                 monthlySales,
                                 all = T,
                                 by = c("Model", "Color", "month")),
                           BHLInPlan,
                           all = T,
                           by = c("Model", "Color", "month")),
                     ProdPlan,
                     all = T,
                     by = c("Model", "Color", "month"))


PlanvActual$monthlyBHLIn[is.na(PlanvActual$monthlyBHLIn)] <- 0
PlanvActual$monthlyProd[is.na(PlanvActual$monthlyProd)] <- 0
PlanvActual$monthlySales[is.na(PlanvActual$monthlySales)] <- 0
PlanvActual$BHLInPlan[is.na(PlanvActual$BHLInPlan)] <- 0
PlanvActual$ProdPlan[is.na(PlanvActual$ProdPlan)] <- 0
PlanvActual$SalesPlan[is.na(PlanvActual$SalesPlan)] <- 0
PlanvActual <- PlanvActual[!is.na(month), ] %>%
  mutate(BHLInGap = monthlyBHLIn - BHLInPlan,
         ProdGap = monthlyProd - ProdPlan,
         SalesGap = monthlySales - SalesPlan)

mc <- select(modelCode, c("PIDI_ModelName", "PIDI_Color", "Sl")) %>%
  rename(Model = PIDI_ModelName, Color = PIDI_Color) %>%
  distinct(Model, Color, .keep_all = T)

PlanvActual <- merge(PlanvActual, mc,
                     all.x = T, all.y = F,
                     by = c("Model", "Color")) %>%
  arrange(Sl, month) %>%
  select(-c("Sl"))

write.csv(PlanvActual, "Output/valueChain/PlanvActual.csv",
          na = "NA")

#Part 11: Stock vs sales plan####

#week wise CKD in (estimated arrival week + 14 days)

ditw <-  rbind(DILoad("Input/DI/Dispatch Information Report 2021.xlsx") %>%
                 select(c("commercialInvoiceNo", "OrderMonth", "PIDI_ModelName",
                          "PIDI_ColorCode", "Units", "DIDate", "BHLOrderMonthYear", "ETAatDestinationPort_Plan", "ETAatDestinationPort_Actual")),
               DILoad("Input/DI/Dispatch Information Report 2022.xlsx") %>%
                 select(c("commercialInvoiceNo", "OrderMonth", "PIDI_ModelName",
                          "PIDI_ColorCode", "Units", "DIDate", "BHLOrderMonthYear", "ETAatDestinationPort_Plan", "ETAatDestinationPort_Actual"))) %>%
  filter(ETAatDestinationPort_Plan > ymd('2022-06-30'), is.na(ETAatDestinationPort_Actual))

ditw <- merge(ditw, DIPILU, all.x = T, all.y = F,
              by = c("PIDI_ModelName", "PIDI_ColorCode")) %>%
  select("commercialInvoiceNo", "PIDI_ModelName", "PIDI_Color", "ETAatDestinationPort_Plan", "Units") %>%
  mutate(BHLinDate = ymd(ETAatDestinationPort_Plan)+14) %>%
  mutate(arrivalYear = year(ymd(BHLinDate)),
         arrivalMonth = month(ymd(BHLinDate)),
         arrivalWeek = ceiling(day(ymd(BHLinDate))/7))

BHLInplan <- ditw[, .(weeklyBHLInPlan = sum(Units, na.rm = T)),
                  by = .(PIDI_ModelName, PIDI_Color, arrivalYear, arrivalMonth, arrivalWeek)]

BHLInThisMonthW1 <- filter(BHLInplan, arrivalYear == year(now()) &
                             arrivalMonth == month(now()) &
                             arrivalWeek == 1) %>%
  rename(BHLInThisMonthW1 = "weeklyBHLInPlan") %>%
  select(-c("arrivalYear", "arrivalMonth", "arrivalWeek"))

BHLInThisMonthW2 <- filter(BHLInplan, arrivalYear == year(now()) &
                             arrivalMonth == month(now()) &
                             arrivalWeek == 2) %>%
  rename(BHLInThisMonthW2 = "weeklyBHLInPlan") %>%
  select(-c("arrivalYear", "arrivalMonth", "arrivalWeek"))

BHLInThisMonthW3 <- filter(BHLInplan, arrivalYear == year(now()) &
                             arrivalMonth == month(now()) &
                             arrivalWeek == 3) %>%
  rename(BHLInThisMonthW3 = "weeklyBHLInPlan") %>%
  select(-c("arrivalYear", "arrivalMonth", "arrivalWeek"))

BHLInThisMonthW4 <- filter(BHLInplan, arrivalYear == year(now()) &
                             arrivalMonth == month(now()) &
                             arrivalWeek == 4) %>%
  rename(BHLInThisMonthW4 = "weeklyBHLInPlan") %>%
  select(-c("arrivalYear", "arrivalMonth", "arrivalWeek"))

BHLInThisMonthW5 <- filter(BHLInplan, arrivalYear == year(now()) &
                             arrivalMonth == month(now()) &
                             arrivalWeek == 5) %>%
  rename(BHLInThisMonthW5 = "weeklyBHLInPlan") %>%
  select(-c("arrivalYear", "arrivalMonth", "arrivalWeek"))

BHLInplan <- merge(merge(merge(merge(BHLInThisMonthW1, BHLInThisMonthW2,
                                           all = T, by = c("PIDI_ModelName", "PIDI_Color")),
                                     BHLInThisMonthW3,
                                     all = T, by = c("PIDI_ModelName", "PIDI_Color")),
                               BHLInThisMonthW4,
                         all = T, by = c("PIDI_ModelName", "PIDI_Color")),
                   BHLInThisMonthW5,
                   all = T, by = c("PIDI_ModelName", "PIDI_Color")) %>%
  rename(Model = "PIDI_ModelName", Color = "PIDI_Color")

#
currentSales <- OrderPiDiBHLInProdSalesNonDuplicate[month(ym(OrderPiDiBHLInProdSalesNonDuplicate$SalesMonth)) == month(now()) &
                                                      year(ym(OrderPiDiBHLInProdSalesNonDuplicate$SalesMonth)) == year(now()),
                                                    .(thisMonthSales = sum(Sales, na.rm = T)),
                                                    by = .(Model, Color)]

currentStock <- OrderPiDiBHLInProdSalesNonDuplicate[, .(TTLCKDStock = sum(CKDStock, na.rm = T),
                                                        TTLCBUStock = sum(CBUStock, na.rm = T)),
                                                    by = .(Model, Color)]


currentSalesPlan <- merge(filter(salesPlan, month(salesPlan$niguriDate) == month(now()) &
                                   year(salesPlan$niguriDate) == year(now())) %>%
                            rename(ThisMonthSalesPlan = "salesQuantity") %>%
                            select(-c("niguriDate")),
                          filter(salesPlan, month(salesPlan$niguriDate) == 1+month(now()) &
                                   year(salesPlan$niguriDate) == year(now())) %>%
                            rename(NextMonthSalesPlan = "salesQuantity") %>%
                            select(-c("niguriDate")),
                          all = T,
                          by = c("NiguriModelName", "NiguriColor"))


currentSalesPlan <- merge(currentSalesPlan, modelCode,
                          all.x = T, all.y = F,
                          by = c("NiguriModelName", "NiguriColor")) %>%
  rename(Model = "PIDI_ModelName", Color = "PIDI_Color") %>%
  select(-c("NiguriModelName", "NiguriColor"))

currentSalesPlan <- currentSalesPlan[!is.na(Model), ]


# currentProdPlan <- filter(prodPlan, month(prodPlan$niguriDate) == month(now()) &
#                             year(prodPlan$niguriDate) == year(now())) %>%
#   rename(ThisMonthProdPlan = "Production_Plan_Quantity") %>%
#   select(-c("niguriDate"))
# 
# 
# currentSalesPlan <- merge(merge(currentSalesPlan, currentProdPlan,
#                                 all = T,
#                                 by = c("NiguriModelName", "NiguriColor")),
#                           modelCode,
#                           all.x = T, all.y = F,
#                           by = c("NiguriModelName", "NiguriColor")) %>%
#   rename(Model = "PIDI_ModelName", Color = "PIDI_Color") %>%
#   select(-c("NiguriModelName", "NiguriColor"))


salesPlanvStock <- merge(merge(merge(currentSalesPlan, currentSales,
                                     all.x = T, all.y = F,
                                     by = c("Model", "Color")),
                               currentStock,
                               all.x = T, all.y = F,
                               by = c("Model", "Color")),
                         BHLInplan,
                         all.x = T, all.y = F,
                         by = c("Model", "Color"))


salesPlanvStock$thisMonthSales[is.na(salesPlanvStock$thisMonthSales)] <- 0

salesPlanvStock$ThisMonthSalesPlan[is.na(salesPlanvStock$ThisMonthSalesPlan)] <- 0

salesPlanvStock$ThisMonthProdPlan[is.na(salesPlanvStock$ThisMonthProdPlan)] <- 0

salesPlanvStock$NextMonthSalesPlan[is.na(salesPlanvStock$NextMonthSalesPlan)] <- 0

salesPlanvStock$thisMonthSales[is.na(salesPlanvStock$thisMonthSales)] <- 0

salesPlanvStock$TTLCKDStock[is.na(salesPlanvStock$TTLCKDStock)] <- 0
salesPlanvStock$TTLCBUStock[is.na(salesPlanvStock$TTLCBUStock)] <- 0
salesPlanvStock$BHLInThisMonthW1[is.na(salesPlanvStock$BHLInThisMonthW1)] <- 0
salesPlanvStock$BHLInThisMonthW2[is.na(salesPlanvStock$BHLInThisMonthW2)] <- 0
salesPlanvStock$BHLInThisMonthW3[is.na(salesPlanvStock$BHLInThisMonthW3)] <- 0
salesPlanvStock$BHLInThisMonthW4[is.na(salesPlanvStock$BHLInThisMonthW4)] <- 0
salesPlanvStock$BHLInThisMonthW5[is.na(salesPlanvStock$BHLInThisMonthW5)] <- 0


salesPlanvStock <- mutate(salesPlanvStock,
                          thisMonthRemainingSalesPlan = ThisMonthSalesPlan - thisMonthSales) %>%
  mutate(CBUShortage = TTLCBUStock - thisMonthRemainingSalesPlan) %>%
#  mutate(prodPlanShortage = CBUShortage - ThisMonthProdPlan) %>%
  mutate(CKDShortage = TTLCKDStock + CBUShortage) %>%
  select(c("Sl", "Model", "Color",
           "ThisMonthSalesPlan", "thisMonthSales", "thisMonthRemainingSalesPlan", "NextMonthSalesPlan",
           "TTLCBUStock", "CBUShortage",
#           "ThisMonthProdPlan", "prodPlanShortage",
           "TTLCKDStock", "CKDShortage",
           "BHLInThisMonthW1", "BHLInThisMonthW2", "BHLInThisMonthW3", "BHLInThisMonthW4", "BHLInThisMonthW5")) %>%
  dplyr::arrange(Sl)

write.csv(salesPlanvStock, "Output/valueChain/salesPlanvStock.csv",
          na = "NA")

rm(BHLInThisMonthW1, BHLInThisMonthW2, BHLInThisMonthW3, BHLInThisMonthW4, BHLInThisMonthW5,
   ditw, currentSalesPlan, currentSales, currentStock)


#Part: Discarded Code####

unsoldCBU <- prodKDP[is.na(prodKDP$salesDate), ] %>%
  filter(prodKDPModelCode == "K0EF-2-B01")

write.csv(unsoldCBU, "Output/unsoldCBU.csv")




## Gap data, calculation and reporting notes####
# PI-DI is for a specific month and is against total MC order, no extra planned values required.
# BHL in gap: plan is 40-60% of order, no need to read. Algorithm: (Order * 0.4/Lot size)*Lot size in Order+3, (1-(Order * 0.4/Lot size)*Lot size ) in Order+4 month.
# 
# Comparison with planned values
# 
# DI: N+2
# 
# BHL IN: N+3: 40%, N+4 60%
#   
# Assumed OK CKD stock: 100% OK
# 
# Assumed production plan: full lot (N+4 40%, N+5 60%)
# 
# Assumed OK CBU stock: 100% OK
# 
# Two approach:
# 1. If it's not possible to break production and sales among order-cycles: month wise (not order cycle wise) gap view for total OK CKD stock, production, CBU stock, sales - against monthly Niguri values
# 
# 2. Ideas to break production and sales among order-cycles: compare in one order-cycle view.
# 
# Assumed sales plan:
# 2.1 Assume sales of produced bikes in that month > problem: production isn't exactly equal to sales plan.
# 2.2 Distribute actual planned sales among order - start at first month and cycles following FIFO (CKD order = sales among month's + next month's CBU carryover)
# Produced amount - sales amount in the latest month = carryover CBU for next cycle.

## Demand gap ideas
# 1. Current demand = drop in dealer MC below dealer stock standard (e.g.15days stock) (how to calculate units?)
# - Nayon San: dealer wise sales summary report




##Branch 2 - add model color serial with bhl in: ideally it should be with order, but I don't have order data now####

# OrderPiDi <- merge(orderPi, monthlyDI,
#                    all.x = T,
#                    all.y = T,
#                    by = c("PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear"))

## Merge BHL in with DI, and montly BHL In


#monthly values - order, pi is month-mode-color wise. Making DI and BHL in similar.

# 
# monthlyBHLIn <- merge(monthlyBHLIn,
#                       PIColorsl, all.x = T, all.y = F,
#                       by = c("PIDI_ModelName", "PIDI_Color")) %>%
#   dplyr::arrange(DIMonthYear, Sl)
# 
# monthlyBHLIn$DiSl <- 1:nrow(monthlyBHLIn)
# 




## Add monthly BHL in, KDP production and KDP sales against monthly DI and MC####
# 
# OrderPiDiBHLInProdSales <- merge(merge(merge(OrderPiDi, monthlyBHLIn,
#                                              all.x = T, all.y = T,
#                                              by = c("DIMonthYear", "PIDI_ModelName", "PIDI_Color")),
#                                        monthlyKDPProd,
#                                        all.x = T, all.y = F,
#                                        by = c("DIMonthYear", "BHLArrivalMonth", "PIDI_ModelName", "PIDI_Color")),
#                                  monthlyKDPSales,
#                                  all.x = T, all.y = F,
#                                  by = c("DIMonthYear", "BHLArrivalMonth", "prodMonth", "PIDI_ModelName", "PIDI_Color")) %>%
#   select(c("PIDI_ModelName", "PIDI_Color",
#            "BHLOrderMonthYear", "OrderQuantity",
#            "PIQuantity",
#            "DiSl", "DIMonthYear", "DIQuantity",
#            "BHLArrivalMonth", "BHLInQuantity",
#            "monthlyProdSl", "prodMonth", "monthlyKDPProd",
#            "salesMonth", "monthlyKDPSales")) %>%
#   dplyr::arrange(DiSl, monthlyProdSl)




# Order PI, DI, BHL In

# 
# OrdervPIvDIvBHLIn <- merge(merge(orderPi, MonthlyDIBHLIn, all.x = T, all.y = F),
#                            PIColorsl, all.x = T, all.y = F,  #merged sl against PI, since there won't be DI against all PIs
#                            by = c("PIDI_ModelName", "PIDI_Color")) %>%
#   filter(!(is.na(OrderQuantity) | OrderQuantity == 0)) %>%
#   #filter(!(is.na(OrderQuantity) & is.na(PIQuantity) & is.na(DIQuantity) & is.na(BHLInQuantity))) %>%
#   #filter(!(OrderQuantity == 0 & PIQuantity == 0 & DIQuantity == 0 & BHLInQuantity == 0)) %>%
#   mutate(OrderVPI = PIQuantity - OrderQuantity,
#          PIVDI = DIQuantity - PIQuantity,
#          OrderVDI = DIQuantity - OrderQuantity,
#          OrderVBHLIn = BHLInQuantity - OrderQuantity,
#          Transit = DIQuantity - BHLInQuantity #,orderToProdDate = my(paste(5+month(BHLOrderMonthYear), year(BHLOrderMonthYear), sep = "-"))
#   ) %>%
#   select(c("Sl", "PIDI_ModelName", "PIDI_Color", "BHLOrderMonthYear", "DIMonthYear",
#            "OrderQuantity", "PIQuantity", "DIQuantity", "BHLInQuantity",
#            "OrderVPI", "OrderVDI", "PIVDI",
#            "OrderVBHLIn", "Transit"))%>%
#   arrange(BHLOrderMonthYear, Sl)
# 






# 
# 
# LUCheck <- select(lookupload, c("ERP Code", "PIDI_ModelName", "Niguri_Status"))
# LUCheck <- LUCheck[!duplicated(`ERP Code`), ]
# LUCheck <- LUCheck[!is.na(`ERP Code`), ] %>%
#   rename(prodKDPModelCode = "ERP Code")
# 
# prodKdpMC <- prodKDP[!duplicated(prodKDPModelCode), c("prodKDPModelCode")]
# prodKdpMC <- prodKdpMC[!is.na(prodKDPModelCode), ]
# 
# not <- merge(prodKdpMC, LUCheck, all.x = T, all.y = F,
#              by = c("prodKDPModelCode"))



# monthwiseProd <- prodKDP[, .(monthlyKDPProd = sum(count, na.rm = T)),
#                          by = .(commercialInvoiceNo, prodMonth, prodKDPModelCode, prodKDPColor)]
# 
# monthwiseSales <- prodKDP[, .(monthlyKDPSales = sum(count, na.rm = T)),
#                           by = .(commercialInvoiceNo, salesMonth, prodKDPModelCode, prodKDPColor)]
# 
# monthlyProdSales <- merge(merge(monthwiseProd, monthwiseSales, all = T),
#                           ,
#                           all.x = T, all.y = F,
#                           by = c("commercialInvoiceNo"))
# monthlyProdSales <- monthlyProdSales[!is.na(DIDate), ]

##Alternate####


##Dummy CKD OK Stock- actual data yet to be collected####

# ckdOKStockLoad <- function(path, sheet, range, monthYear){
#   okckdstock <- data.table(readxl::read_excel(path,
#                                         sheet = sheet,
#                                         range = range)) %>%
#     mutate(OKCKD_Stock_Month = my(monthYear))
#   okckdstock$OKCKD_Stock_Model_Name <- toupper(str_squish(okckdstock$OKCKD_Stock_Model_Name))
#   okckdstock$OKCKD_Stock_Color <- toupper(str_squish(okckdstock$OKCKD_Stock_Color))
#   
#   okckdstock <- okckdstock[!is.na(OKCKD_Stock_Model_Name), ]
#   
#   return(okckdstock)
# }


## Dummy Sales data load####
# 
# salesDataLoad <- function(path, sheet, range, monthYear){
#   salesDataLoad <- data.table(readxl::read_excel(path,
#                                               sheet = sheet,
#                                               range = range)) %>%
#     mutate(OKCKD_Stock_Month = my(monthYear))
#   salesDataLoad$Sales_Model_Name <- toupper(str_squish(salesDataLoad$Sales_Model_Name))
#   salesDataLoad$Sales_Color <- toupper(str_squish(salesDataLoad$Sales_Color))
#   
#   salesDataLoad <- salesDataLoad[!is.na(Sales_Model_Name), ]
#   
#   return(salesDataLoad)
# }




## #internal value chain combine - using KDP production and sales data instead####
# internalValueChain <- merge(merge(merge(merge(okCKDStock, prodPlan, all = T,
#                                               by = c("NiguriModelName", "NiguriColor", "niguriDate")),
#                                         #by.x = c("Production_Plan_Model_Name", "Production_Plan_Color", "niguriDate"),
#                                         #by.y = c("Production_Plan_Model_Name", "Production_Plan_Color", "ProductionPlanDate"))
#                                         OKCBUStock, all = T,
#                                         by = c("NiguriModelName", "NiguriColor", "niguriDate")),
#                                   #by.x = c("ERP_Notification_Model_Name", "niguriDate"),
#                                   #by.y = c("CBUStock_ModelColor", "OKCBUStock_Date"))
#                                   sales, all = T,
#                                   by = c("NiguriModelName", "NiguriColor", "niguriDate")),
#                             internalLU, all.x = T, all.y = F,
#                             by = c("NiguriModelName", "NiguriColor")) %>%
#   select(-c("Production_Plan_Model_Name", "Production_Plan_Color", "ERP_Notification_Model_Name")) %>%
#   filter(!(OKCKDStockQuantity == 0 &
#              Production_Plan_Quantity == 0 &
#              CBU_Quantity == 0 &
#              salesQuantity == 0)) #%>%
# #  mutate(CKDStockDays = (OKCKDStockQuantity / Production_Plan_Quantity) * lubridate::days_in_month(niguriDate),
# #         CBUStockDays = (CBU_Quantity / salesQuantity) * lubridate::days_in_month(niguriDate))
# 

# ##DI-BHL In merge - alt implementation used####
# 
# #monthly DI-BHLIn ####
# DIBHLIn <- merge(di, BHLIn, all.x = T, all.y = F) %>% # merge against invoice number
#   select("BHLOrderDate", "DIDate", "PIDI_ModelName", "PIDI_ColorCode", "Units", "LotSize")
# 
# # Add PIDI_Color column, to take monthly PIDI_Color wise sum
# DIBHLIn <- merge(DIBHLIn, DIPILU, all.x = T, all.y = F,
#                  by = c("PIDI_ModelName", "PIDI_ColorCode"))
# 
# #take sum over order date, di date, model color
# MonthlyDIBHLIn <- DIBHLIn[, .(DIQuantity = sum(Units, na.rm = T),
#                               BHLInQuantity = sum(LotSize, na.rm = T)),
#                           by = .(BHLOrderDate, DIDate, PIDI_ModelName, PIDI_Color)] # month wise summary, so that it can be merged with monthwise pi.



# ## internal and external chain merge - using KDP data for production and sales instead####
# 
# twoChain <- merge(OrdervPIvDIvBHLIn, internalValueChain,
#                   all.x = T, all.y = F,
#                   by.x = c("PIDI_ModelName", "PIDI_Color", "orderToProdDate"),
#                   by.y = c("PIDI_ModelName", "PIDI_Color", "niguriDate")) %>%
#   select(c("Sl", "PIDI_ModelName", "PIDI_Color", "BHLOrderDate", "DIDate",
#            "OrderQuantity", "PIQuantity", "DIQuantity", "BHLInQuantity", "OKCKDStockQuantity",
#            "Production_Plan_Quantity", "CBU_Quantity", "salesQuantity",
#            "OrderVPI", "OrderVDI", "PIVDI", "OrderVBHLIn", "Transit"))%>%
#   arrange(BHLOrderDate, Sl)
# 
# #Update file date
# write.csv(twoChain, "Output/valueChain/20220611twoChain.csv")
# 








### Discarded

# Production is not against BHL in, it's against CKD stock.
# So enter CKD stock layer in between BHL In and Production.

# CKD stock = last month's stock + this month's OK BHL in)
# Carry over = last month's CKD Stock - Last month's production
# This month's (total CKD stock - production) will be next month's starting CKD Stock.

# Load OK CKD stock for the first month. Subsequent month's OK CKD stock should be from data load.

