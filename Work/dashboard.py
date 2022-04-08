import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NOTES: Recreate pivot tables and chart with matplotlib


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
    dataset.drop(index=dataset[dataset["SEGMENT"] == "КТП ЧАСТНО БАНКИРАНЕ"].index, inplace=True)
    return dataset


def applications_segment(dataset):   # Requires columns named "SUB_PRODUCT" & "WORKINGPLACE". Produces column "SEGMENT"
    segment_conditions = [
        (dataset["SUB_PRODUCT"] == "KTП Стандартен") & (dataset["WORKINGPLACE"] == "НОИ")]
    segment_values = ["Пенсионер"]
    dataset["SEGMENT"] = np.select(segment_conditions, segment_values, default=dataset["SUB_PRODUCT"])
    dataset["SEGMENT"] = dataset["SEGMENT"].str.upper()
    dataset.drop(index=dataset[dataset["SEGMENT"] == "КТП ЧАСТНО БАНКИРАНЕ"].index, inplace=True)
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

customers = pd.read_csv(r"C:\Yanko\Customers_list_GDPR.csv", sep=";", dtype={"CUSTOMER_ID": str, "REGION(ETP)/NTP": str,
                                                                             "TEST/CONTROL": str, "ETP/NTP": str})
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

# BRANCHES LIST & CUTOMERS TABLE PREPARATION

# In order to match values in applications table BRANCH_NAME is set to capital letters
branches["BRANCH_NAME"] = branches["BRANCH_NAME"].str.upper()
customers.rename(columns={"CUSTOMER_ID": "CUST_ID"}, inplace=True)

# ----------------------------------------------------------------------------------------------------------------------

# EXCEPTIONS TABLE PREPARATION
exceptions.rename(columns={"1": "CUST_ID", "27": "APPROVED_IR"}, inplace=True)
exceptions["APPROVED_IR"] *= 100

# ----------------------------------------------------------------------------------------------------------------------

# CONSULTATIONS MANIPULATION

# Rename customer number column to match other tables, adding ETP/NTP & TEST/CONTROL tags, dropping not found clients
consultations.rename(columns={"CUSTOMERNUMBER": "CUST_ID"}, inplace=True)
consultations = consultations.merge(customers[["CUST_ID", "ETP/NTP", "TEST/CONTROL"]], how="left", on="CUST_ID")
consultations.dropna(subset=["TEST/CONTROL"], inplace=True)

# Tagging pilot included loans in PRODUCT_CATEGORY as "OK" and selecting only them
consultations["PRODUCT_CATEGORY"].replace(["СТАНДАРТЕН КРЕДИТ ЗА ТЕКУЩО ПОТРЕБЛЕНИЕ"], "OK", inplace=True)
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
applications.rename(columns={"KBICUSTNO": "CUST_ID"}, inplace=True)
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

contacted_customers["SEGMENT"] = contacted_customers["SEGMENT"].replace("КТП СТАНДАРТЕН", "KTП СТАНДАРТЕН")

# Merging both tables with contacted clients and removing duplicates based on customer ID
contacted_customers = pd.concat([contacted_customers, temp_applications])
contacted_customers.drop_duplicates(["CUST_ID"], inplace=True)

contacted_customers.to_csv(r"C:\Yanko\Result\Contacted_customers_result.csv", index=False, sep=";", encoding="utf-8-sig")

# ----------------------------------------------------------------------------------------------------------------------

# Clients without pilot consultation (based on product index value)

pilot_consultations = consultations[consultations["PRODUCT_INDEX"] >= 80000000].copy()
pilot_consultations.drop_duplicates(["CUST_ID"], inplace=True)
non_pilot_consultations = pd.merge(consultations, pilot_consultations, how="outer", on="CUST_ID", indicator=True)
non_pilot_consultations.drop(index=non_pilot_consultations[non_pilot_consultations["_merge"] == "both"].index, inplace=True)
non_pilot_consultations.drop_duplicates(["CUST_ID"], inplace=True)
clients_without_pilot_consultation = non_pilot_consultations[["CUST_ID", "CONS_CREATE_DATE_x", "BRANCH_ID_x", "BRANCH_x"]].copy()
clients_without_pilot_consultation.to_csv(r"C:\Yanko\Result\Clients_without_pilot_consultation.csv", index=False, sep=";", encoding="utf-8-sig")

# ----------------------------------------------------------------------------------------------------------------------

# PIVOT TABLES CREATION AND CHARTS/PLOTS

# APPROVED LOANS PIVOT TABLES

# DISTRIBUTION by CHANNEL

distribution_of_apps = applications[["TEST/CONTROL", "PILOT_CHANNEL"]]
distribution_pt = pd.crosstab(index=distribution_of_apps["TEST/CONTROL"], columns=distribution_of_apps["PILOT_CHANNEL"], normalize="index")
distribution_pt = distribution_pt[["MODEL/IBP", "PROMO", "EXCEPTION", "OUTSIDE_PROCESS"]]

distribution_pt.plot(kind="barh", edgecolor="black", stacked=True, colormap="gist_rainbow", figsize=(10, 6))
plt.legend(loc="upper left", ncol=4)
plt.xlabel("Share of sales (by #)")
plt.ylabel("Test/Control")

for n, x in enumerate([*distribution_pt.index.values]):
    for (proportion, y_loc) in zip(distribution_pt.loc[x], distribution_pt.loc[x].cumsum()):
        plt.text(y=n-0.17, x=(y_loc - proportion) + (proportion / 2), s=f"{int(np.round(proportion * 100, 0))}%", color="black",
                 fontsize=12, fontweight="bold")

plt.show()

# CONTACTED CUSTOMERS by TEST/CONTROL
contacted_customers_pt_test_control = pd.pivot_table(contacted_customers, values=["CUST_ID"], index=["TEST/CONTROL"], columns=[],
                                                     aggfunc={"CUST_ID": np.count_nonzero})
contacted_customers_pt_test_control.sort_values(by="TEST/CONTROL", ascending=False, inplace=True)

# MODEL SALES by TEST/CONTROL

model_applications = applications[applications["PILOT_CHANNEL"] == "MODEL/IBP"]

air_func = lambda x: np.average(x, weights=model_applications.loc[x.index, "AMOUNT"])
applications_pt = pd.pivot_table(model_applications, values=["CUST_ID", "AMOUNT", "NPV", "INTERESTRATE"], index=["TEST/CONTROL"], columns=[],
                         aggfunc={"CUST_ID": np.count_nonzero, "AMOUNT": np.mean, "NPV": np.sum, "INTERESTRATE": air_func})
applications_pt.sort_values(by="TEST/CONTROL", ascending=False, inplace=True)

report_pt = contacted_customers_pt_test_control.merge(applications_pt[["CUST_ID", "AMOUNT", "NPV", "INTERESTRATE"]], how="left", on="TEST/CONTROL")
report_pt["NPV"] = report_pt["NPV"] / report_pt["CUST_ID_x"]
report_pt.reset_index(inplace=True)
report_pt.rename(columns={"AMOUNT": "TICKET_SIZE"}, inplace=True)
print(report_pt)


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha="center")

# CHART

plot_data = report_pt
plot_data["TICKET_SIZE"] = (plot_data["TICKET_SIZE"] / 1000).__round__(1)
plot_data["INTERESTRATE"] = plot_data["INTERESTRATE"].__round__(2)

fig, ax1 = plt.subplots(figsize=(8,5))
x = report_pt["TEST/CONTROL"]
y1 = report_pt["TICKET_SIZE"]
y2 = report_pt["INTERESTRATE"]

ax1.bar(x, y1, width=0.5, alpha=0.5, color="green", edgecolor="black")
ax1.grid(False)
ax1.set_ylabel("TICKET_SIZE")
ax1.set_ylim(15, 19)
ax1.legend(["TICKET_SIZE"], loc="upper right")
addlabels(x, y1)

ax2 = ax1.twinx()

ax2.plot(x, y2)
ax2.set_ylabel("AIR")
ax2.set_ylim(4.0, 6.5)
ax2.legend(["AIR"], loc="upper left")
addlabels(x, y2)

plt.show()

# CONTACTED CUSTOMERS by TEST/CONTROL & SEGMENT
contacted_customers_pt_segment = contacted_customers.pivot_table(values=["CUST_ID"], index=["TEST/CONTROL", "SEGMENT"], columns=[],
                                                     aggfunc={"CUST_ID": np.count_nonzero})
contacted_customers_pt_segment.sort_values(by="TEST/CONTROL", ascending=False, inplace=True)
print(contacted_customers_pt_segment)

# MODEL SALES by TEST/CONTROL & SEGMENT

model_applications = applications[applications["PILOT_CHANNEL"] == "MODEL/IBP"]

applications_pt_segment = pd.pivot_table(model_applications, values=["CUST_ID", "AMOUNT", "NPV", "INTERESTRATE"], index=["TEST/CONTROL",
                "SEGMENT"], columns=[], aggfunc={"CUST_ID": np.count_nonzero, "AMOUNT": np.mean, "NPV": np.sum, "INTERESTRATE": air_func})
applications_pt_segment.sort_values(by="TEST/CONTROL", ascending=False, inplace=True)

report_pt_segment = contacted_customers_pt_segment.merge(applications_pt_segment[["CUST_ID", "AMOUNT", "NPV", "INTERESTRATE"]],
                                                         how="left", on=["TEST/CONTROL", "SEGMENT"])

report_pt_segment["NPV"] = report_pt_segment["NPV"] / report_pt_segment["CUST_ID_x"]
report_pt_segment.reset_index(inplace=True)
report_pt_segment.rename(columns={"AMOUNT": "TICKET_SIZE"}, inplace=True)
report_pt_segment.plot()

# CHART / TABLE

fig_2, ax = plt.subplots()
fig_2.patch.set_visible(False)
ax.axis("off")
ax.axis("tight")
#test
ax.table(cellText=report_pt_segment.values, colLabels=report_pt_segment.columns, loc="center")
fig_2.tight_layout()
plt.show()

# plot_data_2 = report_pt_segment
# plot_data_2["TICKET_SIZE"] = (plot_data_2["TICKET_SIZE"] / 1000).__round__(1)
# plot_data_2["INTERESTRATE"] = plot_data_2["INTERESTRATE"].__round__(2)
#
# fig_2, ax1 = plt.subplots(figsize=(8,5))
# x = report_pt_segment["SEGMENT"]
# y1 = report_pt_segment["TICKET_SIZE"]
# y2 = report_pt_segment["INTERESTRATE"]
#
# ax1.bar(x, y1, width=0.5, alpha=0.5, color="green", edgecolor="black")
# ax1.bar(x, y1, width=-0.5, alpha=0.5, color="red", edgecolor="black")
# ax1.grid(False)
# ax1.set_ylabel("TICKET_SIZE")
# ax1.set_ylim(15, 19)
# ax1.legend(["TICKET_SIZE"], loc="upper right")
# addlabels(x, y1)
#
# ax2 = ax1.twinx()
#
# ax2.plot(x, y2)
# ax2.set_ylabel("AIR")
# ax2.set_ylim(4.0, 6.5)
# ax2.legend(["AIR"], loc="upper left")
# addlabels(x, y2)
#
# plt.show()
