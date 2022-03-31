import numpy as np
import pandas as pd

# NOTES:


def income_range_calc(dataset):  # Requires column named "AVERAGE_INCOME". Produces column with name "INCOME_RANGE"
    dataset_income_range_conditions = [
        (dataset["AVERAGE_INCOME"] < 750),
        (dataset["AVERAGE_INCOME"] >= 750) & (dataset["AVERAGE_INCOME"] < 1500),
        (dataset["AVERAGE_INCOME"] >= 1500) & (dataset["AVERAGE_INCOME"] < 2500),
        (dataset["AVERAGE_INCOME"] >= 2500) & (dataset["AVERAGE_INCOME"] < 4000),
        (dataset["AVERAGE_INCOME"] >= 4000)]
    dataset_income_range_values = ["Up to BGN 750", "BGN 750 - 1 500", "BGN 1 500 - 2 500", "BGN 2 500 - 4 000",
                                   "Above BGN 4 000"]
    dataset["INCOME_RANGE"] = np.select(dataset_income_range_conditions, dataset_income_range_values)
    return dataset


def size_range_calc(dataset):  # Requires column named "AMOUNT". Produces column with name "BELOW/ABOVE_30k"
    size_range_conditions = [
        (dataset["AMOUNT"] < 30000),
        (dataset["AMOUNT"] >= 30000)]
    size_range_values = ["30-", "30+"]
    dataset["BELOW/ABOVE_30k"] = np.select(size_range_conditions, size_range_values)
    return dataset


def consultation_segment(dataset):  # Requires column named "PROGRAM". Produces column with name "SEGMENT"
    dataset["PROGRAM"] = dataset["PROGRAM"].str.upper()
    segment_conditions = [
        (dataset["PROGRAM"].str.contains("ПРЕСТИЖ")),
        (dataset["PROGRAM"].str.contains("ПРАКТИКА")),
        (dataset["PROGRAM"].str.contains("ЛИДЕР")),
        (dataset["PROGRAM"].str.contains("ПРЕМИУМ")),
        (dataset["PROGRAM"].str.contains("ПЕНСИЯ"))]
    segment_values = ["КТП ПРЕСТИЖ ПЛЮС", "КТП ПРАКТИКА", "КТП ЛИДЕР", "КТП ЧАСТНО БАНКИРАНЕ", "ПЕНСИОНЕР"]
    dataset["SEGMENT"] = np.select(segment_conditions, segment_values)
    dataset["SEGMENT"] = dataset["SEGMENT"].replace("0", "КТП СТАНДАРТЕН")
    return dataset


def applications_segment(dataset):   # Requires columns named "SUB_PRODUCT" & "WORKINGPLACE". Produces column "SEGMENT"
    segment_conditions = [
        (dataset["SUB_PRODUCT"] == "KTП Стандартен") & (dataset["WORKINGPLACE"] == "НОИ")]
    segment_values = ["Пенсионер"]
    dataset["SEGMENT"] = np.select(segment_conditions, segment_values, default=dataset["SUB_PRODUCT"])
    dataset["SEGMENT"] = dataset["SEGMENT"].str.upper()
    return dataset


def applications_channel(dataset):
    dataset = dataset.merge(exceptions[["CUST_ID", "APPROVED_IR"]], how="left", on="CUST_ID")

    channel_conditions = [
        (dataset["PRODUCTINDEX"] >= 80000000),
        (dataset["INTERESTRATE"] == 4.77) | (dataset["INTERESTRATE"] == 4.88) | (dataset["INTERESTRATE"] == 4.99),
        (dataset["INTERESTRATE"] == dataset["APPROVED_IR"])]
    channel_values = ["MODEL/IBP", "PROMO", "EXCEPTION"]

    dataset["PILOT_CHANNEL"] = np.select(channel_conditions, channel_values, default="OUTSIDE_PROCESS")

    dataset.sort_values(["PILOT_CHANNEL"], inplace=True)
    dataset.drop_duplicates(["CREDITREQID"], inplace=True)
    dataset.drop(["APPROVED_IR"], axis=1, inplace=True)
    return dataset


def npv_calc(dataset):
    # dataset["INTERESTRATE_2"] = round(dataset["INTERESTRATE"] / 100, 3)
    # dataset = dataset.astype({"INTERESTRATE_2": str})
    #
    # dataset["NPV"] = npv_table.at[dataset["CREDITDURATION"], dataset["INTERESTRATE_2"]]

    npv = list()
    interest_rate_list = dataset["INTERESTRATE"].tolist()
    interest_rate_list = [round(x / 100, 3) for x in interest_rate_list]
    remaining_term_list = dataset["CREDITDURATION"].tolist()
    amount_list = dataset["AMOUNT"].tolist()

    for i in range(len(amount_list)):
        loan_npv = npv_table._get_value(min(int(remaining_term_list[i]), 120), str(interest_rate_list[i]))
        loan_npv *= amount_list[i]
        npv.append(loan_npv)
    dataset["NPV"] = npv
    return dataset


# READING CSV FILES____________________________________________________________________________________________________

customers = pd.read_csv(r"C:\Yanko\Customers_list_GDPR.csv", sep=";", dtype={"CUST_ID": str, "REGION(ETP)/NTP": str,
                                                                             "TEST/CONTROL": int, "ETP/NTP": str})
branches = pd.read_csv(r"C:\Yanko\Branches_list.csv", encoding="utf-8-sig", sep=";", dtype={"BRANCH_ID": str,
                                                                                            "BRANCH_NAME": str})
exceptions = pd.read_csv(r"C:\Yanko\Exceptions.csv", encoding="utf-8-sig", sep=";", dtype={"1": str,
                                                                                                     "27": float})
applications = pd.read_csv(r"C:\Yanko\Applications_GDPR.csv", encoding="utf-8-sig", sep=";",
                           dtype={"KBICUSTNO": str, "AMOUNT": float, "PRODUCT_INDEX": float, "CREDITDURATION": int},
                           date_parser=["APPLICATION_DATE"])
consultations = pd.read_csv(r"C:\Yanko\Consultations_GDPR.csv", encoding="utf-8-sig", sep=";", na_values=" ",
                            dtype={"CUSTOMERNUMBER": str, "DML_BRN_0_CODE": str, "FIRSTMONTHINCOME": float,
                                   "SECONDMONTHINCOME": float, "THIRDMONTHINCOME": float, "OFFER_AMNT": float,
                                   "OFFERID": float}, parse_dates=["CONS_CREATE_DATE"], dayfirst=True)
npv_table = pd.read_csv(r"C:\Yanko\NPV_tables_NEW_loans.csv", sep=";", dtype={"MONTH": int})

# ----------------------------------------------------------------------------------------------------------------------

# BRANCHES LIST TABLE PREPARATION

# In order to match values in applications table BRANCH_NAME is set to capital letters
branches["BRANCH_NAME"] = branches["BRANCH_NAME"].str.upper()

# ----------------------------------------------------------------------------------------------------------------------

# EXCEPTIONS TABLE PREPARATION
exceptions = exceptions.rename(columns={"1": "CUST_ID"})
exceptions = exceptions.rename(columns={"27": "APPROVED_IR"})
exceptions["APPROVED_IR"] *= 100

# ----------------------------------------------------------------------------------------------------------------------

# CONSULTATIONS MANIPULATION

# Rename customer number column to match other tables, adding ETP/NTP & TEST/CONTROL tags, dropping not found clients
consultations = consultations.rename(columns={"CUSTOMERNUMBER": "CUST_ID"})
consultations = consultations.merge(customers[["CUST_ID", "ETP/NTP", "TEST/CONTROL"]], how="left", on="CUST_ID")
consultations.dropna(subset=["TEST/CONTROL"], inplace=True)

# Tagging pilot included loans in PRODUCT_CATEGORY as "OK" and selecting only them
consultations["PRODUCT_CATEGORY"] = consultations["PRODUCT_CATEGORY"].replace(["СТАНДАРТЕН КРЕДИТ ЗА ТЕКУЩО ПОТРЕБЛЕНИЕ"], "OK")
consultations = consultations[consultations["PRODUCT_CATEGORY"] == "OK"]

# Remove rows not matched with pilot branches list
consultations.rename(columns={"DML_BRN_0_CODE": "BRANCH_ID"}, inplace=True)
consultations = consultations[consultations["BRANCH_ID"].isin(branches["BRANCH_ID"])]

# Creating additional columns (AVERAGE_INCOME, INCOME_RANGE, BELOW/ABOVE_30k, SEGMENT, WEEK)

consultations["AVERAGE_INCOME"] = consultations[["FIRSTMONTHINCOME", "SECONDMONTHINCOME",
                                                 "THIRDMONTHINCOME"]].mean(axis=1)
consultations = income_range_calc(consultations)  # INCOME RANGE

consultations = consultations.rename(columns={"OFFER_AMNT": "AMOUNT"})
consultations = size_range_calc(consultations)  # ABOVE/BELOW_30k

consultations = consultation_segment(consultations)  # Creating SEGMENT column

consultations["CONS_CREATE_DATE"] = pd.to_datetime(consultations["CONS_CREATE_DATE"])
consultations["WEEK"] = consultations["CONS_CREATE_DATE"].dt.week  # WEEK of consultation

# Sorting values in descending order by OFFER_ID
consultations.sort_values(["OFFERID"], ascending=False, inplace=True)

consultations.to_csv(r"C:\Yanko\Result\Consultations_result.csv", index=True, sep=";", encoding="utf-8-sig")

# ----------------------------------------------------------------------------------------------------------------------

# APPLICATIONS MANIPULATION


applications = applications[applications["DECISION"] == "Approved"]  # Filtering rows with "Approved" decision

# Rename customer number column to match other tables, adding ETP/NTP & TEST/CONTROL tags, dropping not found clients
applications = applications.rename(columns={"KBICUSTNO": "CUST_ID"})
applications = applications.merge(customers[["CUST_ID", "ETP/NTP", "TEST/CONTROL"]], how="left", on="CUST_ID")
applications.dropna(subset=["TEST/CONTROL"], inplace=True)

# Remove rows not matched with pilot branches list
applications = applications[applications["BRANCH_NAME"].isin(branches["BRANCH_NAME"])]

# Remove applications with sub_product "КТП Студентски" and sailors (by PRODUCT INDEX)
applications.drop(index=applications[applications["SUB_PRODUCT"] == "КТП Студентски"].index, inplace=True)
applications.drop(index=applications[applications["PRODUCTINDEX"] == 40727].index, inplace=True)

# Adding additional columns (AVERAGE_INCOME, INCOME_RANGE, BELOW/ABOVE_30k,
# PENSIONER SEGMENT, EXCEPTIONS, WEEK, NPV, IR_PRODUCT)

applications = applications.rename(columns={"INCOME": "AVERAGE_INCOME"})
applications = income_range_calc(applications)  # INCOME RANGE col

applications = size_range_calc(applications)  # BELOW/ABOVE_30k col

applications = applications_segment(applications)  # Adding PENSIONER SEGMENT

applications = applications_channel(applications)  # Adding CHANNEL col (MODEL/IBP, PROMO, EXCEPTION, OUTSIDE_PROCESS)

applications["APPLICATION_DATE"] = pd.to_datetime(applications["APPLICATION_DATE"])
applications["WEEK"] = applications["APPLICATION_DATE"].dt.week  # Adding column WEEK to applications

applications = npv_calc(applications)  # Adding NPV value col

applications["IR_AMOUNT_PRODUCT"] = applications["INTERESTRATE"] * applications["AMOUNT"]  # Add product col for AIR

applications.to_csv(r"C:\Yanko\Result\Applications_result.csv", index=False, sep=";", encoding="utf-8-sig")

# ----------------------------------------------------------------------------------------------------------------------

# CONTACTED CUSTOMER TABLE CREATION

# Selecting pilot clients from consultation table
contacted_customers = consultations[["CUST_ID", "ETP/NTP", "TEST/CONTROL", "INCOME_RANGE", "BELOW/ABOVE_30k",
                                     "SEGMENT", "WEEK"]].copy()

# Selecting contacted clients from application
temp_applications = applications[["CUST_ID", "ETP/NTP", "TEST/CONTROL", "INCOME_RANGE", "BELOW/ABOVE_30k",
                                  "SEGMENT", "WEEK"]].copy()

# Merging both tables with contacted clients and removing duplicates based on customer ID
contacted_customers = pd.concat([contacted_customers, temp_applications])
contacted_customers.drop_duplicates(["CUST_ID"], inplace=True)

contacted_customers.to_csv(r"C:\Yanko\Result\Contacted_customers_result.csv", index=False, sep=";", encoding="utf-8-sig")

# ----------------------------------------------------------------------------------------------------------------------
