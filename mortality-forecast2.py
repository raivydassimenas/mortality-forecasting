import datetime
import math

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# import plotly

import numpy as np
import pandas as pd

# import sklearn
# import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.layers import RNN, GRU, LSTM, GRUCell, LSTMCell, Dense
# import keras

# from cond_rnn import ConditionalRecurrent

TRAINING_SIZE = 49  # number of observations used for training
WINDOW = 3  # the length of the sequence used for prediction
CONDITIONAL_RNN = False  # whether we use conditional RNN (by gender and country)

tf.random.set_seed(1918)


# Data preprocessing


def read_data(filenames):
    dataframes = []
    for filename in filenames:
        dataframes.append(pd.read_csv(filename, header=1, sep=r"\s+"))
    return dataframes


def long_to_wide(dataset):
    dataset = dataset.melt(id_vars=["Year", "Age"], value_vars=["Male", "Female"])
    dataset = dataset.pivot(
        columns=["Age"], index=["Year", "variable"], values=["value"]
    )
    dataset = dataset.reset_index()
    dataset["variable"] = dataset["variable"] == "Female"
    dataset = dataset.astype("float")
    dataset["Year"] = dataset["Year"].transform(lambda x: datetime.date(int(x), 1, 1))
    dataset.columns = dataset.columns.get_level_values(1)
    dataset.columns.values[0] = "Year"
    dataset.columns.values[1] = "F"
    dataset = dataset.set_index(dataset.columns[0])
    dataset = dataset.astype("float")
    dataset = dataset.sort_values(by=["F", "Year"])
    return dataset


def clean_data(dataframes):
    clean_dataframes = []
    for dataframe in dataframes:
        dataframe = dataframe.drop(columns="Total")
        dataframe = long_to_wide(dataframe)
        clean_dataframes.append(dataframe)
    return clean_dataframes


(
    lt_mortality_raw,
    lt_deaths_raw,
    lt_exposure_raw,
    lv_mortality_raw,
    lv_deaths_raw,
    lv_exposure_raw,
    ee_mortality_raw,
    ee_deaths_raw,
    ee_exposure_raw,
) = read_data(
    (
        "./data/LT_Mx_5x1.txt",
        "./data/LT_Deaths_5x1.txt",
        "./data/LT_Exposures_5x1.txt",
        "./data/LV_Mx_5x1.txt",
        "./data/LV_Deaths_5x1.txt",
        "./data/LV_Exposures_5x1.txt",
        "./data/EE_Mx_5x1.txt",
        "./data/EE_Deaths_5x1.txt",
        "./data/EE_Exposures_5x1.txt",
    )
)

lt_mortality, lv_mortality, ee_mortality = (
    lt_mortality_raw.replace(".", None),
    lv_mortality_raw.replace(".", None),
    ee_mortality_raw.replace(".", None),
)

lt_mortality, lt_deaths, lt_exposure = clean_data(
    (lt_mortality, lt_deaths_raw, lt_exposure_raw)
)

lv_mortality, lv_deaths, lv_exposure = clean_data(
    (lv_mortality, lv_deaths_raw, lv_exposure_raw)
)

ee_mortality, ee_deaths, ee_exposure = clean_data(
    (ee_mortality, ee_deaths_raw, ee_exposure_raw)
)

lt_mortality["LT"], lv_mortality["LV"], ee_mortality["EE"] = 1, 1, 1

lt_deaths, lt_mortality, lt_exposure = (
    lt_deaths.drop(lt_mortality.tail(1).index),
    lt_mortality.drop(lt_mortality.tail(1).index),
    lt_exposure.drop(lt_mortality.tail(1).index),
)


# Grouping mortality into wider intervals


def sum_groups(dataframe, combined_groups, summing_groups):
    for idx in range(len(combined_groups)):
        dataframe[combined_groups[idx]] = dataframe.loc[:, summing_groups[idx]].sum(
            axis=1
        )
    return dataframe


def combine_mortality(
    mortality_df, deaths_df, exposure_df, combined_groups, summing_groups
):
    for i in range(len(combined_groups)):
        mortality_df[combined_groups[i]] = (
            deaths_df[combined_groups[i]] / exposure_df[combined_groups[i]]
        )
        mortality_df = mortality_df.drop(columns=summing_groups[i])
    return mortality_df


combined_groups = ["1-9"]
summing_groups = [["1-4", "5-9"]]

for i in range(1, 9):
    combined_groups.append(str(i) + "0-" + str(i) + "9")
combined_groups.append("90+")

for i in range(1, 9):
    summing_groups.append([str(i) + "0-" + str(i) + "4", str(i) + "5-" + str(i) + "9"])
summing_groups.append(["90-94", "95-99", "100-104", "105-109", "110+"])

combined_groups.insert(0, "0")
summing_groups.insert(0, ["0"])
x = combined_groups

lt_deaths, lt_exposure = (
    sum_groups(lt_deaths, combined_groups, summing_groups),
    sum_groups(lt_exposure, combined_groups, summing_groups),
)

lv_deaths, lv_exposure = (
    sum_groups(lv_deaths, combined_groups, summing_groups),
    sum_groups(lv_exposure, combined_groups, summing_groups),
)

ee_deaths, ee_exposure = (
    sum_groups(ee_deaths, combined_groups, summing_groups),
    sum_groups(ee_exposure, combined_groups, summing_groups),
)

lt_mortality, lv_mortality, ee_mortality = (
    combine_mortality(
        lt_mortality, lt_deaths, lt_exposure, combined_groups, summing_groups
    ),
    combine_mortality(
        lv_mortality, lv_deaths, lv_exposure, combined_groups, summing_groups
    ),
    combine_mortality(
        ee_mortality, ee_deaths, ee_exposure, combined_groups, summing_groups
    ),
)


# Data separation into training and test


def create_dataset(dataset, length):
    predictors, response, conditions = [], [], []

    for i in range(len(dataset) - length):
        value = dataset.iloc[i : (i + length)]
        if (
            np.unique(value.iloc[:, 0]).size
            * np.unique(value.iloc[:, 1]).size
            * np.unique(value.iloc[:, 2]).size
            * np.unique(value.iloc[:, 3]).size
            == 1
        ) and dataset.iloc[i : i + length + 1, :].index.is_monotonic_increasing:
            predictors.append(value.iloc[:, 4:])
            response.append(dataset.iloc[i + length, 4:])
            conditions.append(
                [value.iloc[0, 0], value.iloc[0, 1], value.iloc[0, 2], value.iloc[0, 3]]
            )

    return np.array(predictors), np.array(response), np.array(conditions)
    # return tf.convert_to_tensor(predictors), tf.convert_to_tensor(response), tf.convert_to_tensor(conditions)


dataset = pd.concat([lt_mortality, lv_mortality, ee_mortality])

dataset[["LT", "LV", "EE"]] = dataset[["LT", "LV", "EE"]].fillna(0)

dataset.insert(0, "LT", dataset.pop("LT"))
dataset.insert(1, "LV", dataset.pop("LV"))
dataset.insert(2, "EE", dataset.pop("EE"))

scaler = MinMaxScaler(feature_range=(0, 1))
dataset.iloc[:, 4:] = scaler.fit_transform(dataset.iloc[:, 4:])

train = dataset[np.mod(np.arange(dataset.shape[0]), 61) < TRAINING_SIZE]
# test = pd.concat([train, dataset]).drop_duplicates(keep=False)
test = dataset[np.mod(np.arange(dataset.shape[0]), 61) >= TRAINING_SIZE]

# Neural network training

train_predictors, train_response, train_conditions = create_dataset(train, WINDOW)
test_predictors, test_response, test_conditions = create_dataset(test, WINDOW)
train_predictors = train_predictors.transpose(0, 2, 1)
test_predictors = test_predictors.transpose(0, 2, 1)

print(f'dataset: {dataset.head()}')
print(f'train_predictors.shape: {train_predictors.shape}')
print(f'train_conditions: {train_conditions[:5]}')