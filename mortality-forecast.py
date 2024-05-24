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
test = pd.concat([train, dataset]).drop_duplicates(keep=False)

# Neural network training

train_predictors, train_response, train_conditions = create_dataset(train, WINDOW)
test_predictors, test_response, test_conditions = create_dataset(test, WINDOW)
train_predictors = train_predictors.transpose(0, 2, 1)
test_predictors = test_predictors.transpose(0, 2, 1)


def create_model(model_type, units, hidden_layer_units):
    model_to_create = tf.keras.Sequential()
    if model_type == "GRU":
        if CONDITIONAL_RNN:
            model_to_create.add(RNN(GRUCell(units)))
        else:
            model_to_create.add(GRU(units))
    elif model_type == "LSTM":
        if CONDITIONAL_RNN:
            model_to_create.add(RNN(LSTMCell(units)))
        else:
            model_to_create.add(LSTM(units))

    if hidden_layer_units > 0:
        model_to_create.add(Dense(hidden_layer_units))
    model_to_create.add(Dense(10))

    model_to_create.compile(loss="mean_squared_error", optimizer="adam")
    return model_to_create


def train_model(model_to_train):
    model_to_train.fit(
        x=[train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors,
        y=train_response,
        epochs=5,  # 500,
        batch_size=1,
    )
    return model_to_train


def get_results(model_type):
    results = []
    models = []
    for UNITS in range(3, 5):
        for HIDDEN_LAYER_UNITS in range(3, 5):
            results.append([UNITS, HIDDEN_LAYER_UNITS])

            # train_predictors, train_response, train_conditions = create_dataset(
            #     train, WINDOW)
            # test_predictors, test_response, test_conditions = create_dataset(
            #     test, WINDOW)
            # train_predictors = train_predictors.transpose(0, 2, 1)
            # test_predictors = test_predictors.transpose(0, 2, 1)

            model = create_model(model_type, UNITS, HIDDEN_LAYER_UNITS)
            model = train_model(model)
            models.append(model)

            train_prediction = model(
                [train_predictors, train_conditions]
                if CONDITIONAL_RNN
                else train_predictors
            )
            test_prediction = model(
                [test_predictors, test_conditions]
                if CONDITIONAL_RNN
                else test_predictors
            )

            results.append(
                math.sqrt(
                    mean_squared_error(
                        scaler.inverse_transform(train_response),
                        scaler.inverse_transform(train_prediction),
                    )
                )
            )
            results.append(
                math.sqrt(
                    mean_squared_error(
                        scaler.inverse_transform(test_response),
                        scaler.inverse_transform(test_prediction),
                    )
                )
            )
    return results, models


# res = []
# gru_models = []

res, gru_models = get_results("GRU")

# for GRU_UNITS in range(3, 5):
#     for HIDDEN_LAYER_UNITS in range(3, 5):
#         res.append([GRU_UNITS, HIDDEN_LAYER_UNITS])
#
#         train_predictors, train_response, train_conditions = create_dataset(
#             train, WINDOW)
#         test_predictors, test_response, test_conditions = create_dataset(
#             test, WINDOW)
#         train_predictors = train_predictors.transpose(0, 2, 1)
#         test_predictors = test_predictors.transpose(0, 2, 1)
#
#         # model = tf.keras.Sequential()
#         # if CONDITIONAL_RNN:
#         #     model.add(ConditionalRecurrent(GRU(GRU_UNITS)))
#         # else:
#         #     model.add(GRU(GRU_UNITS))
#         #
#         # if HIDDEN_LAYER_UNITS > 0:
#         #     model.add(layers.Dense(HIDDEN_LAYER_UNITS))
#         # model.add(layers.Dense(11))
#         #
#         # model.compile(loss='mean_squared_error', optimizer='adam')
#
#         model = create_model(GRU_UNITS, HIDDEN_LAYER_UNITS)
#         model = train_model(model, train_predictors, train_response, train_conditions)
#
#         # model.fit(
#         #     x=[train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors, \
#         #     y=train_response, epochs=500, batch_size=1)
#
#         train_prediction = model([train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors)
#         test_prediction = model([test_predictors, test_conditions] if CONDITIONAL_RNN else test_predictors)
#         res.append(math.sqrt(
#             mean_squared_error(scaler.inverse_transform(train_response), scaler.inverse_transform(train_prediction))))
#         res.append(math.sqrt(
#             mean_squared_error(scaler.inverse_transform(test_response), scaler.inverse_transform(test_prediction))))
#         gru_models.append(model)

np.savetxt("gru.txt", np.asarray(res), fmt="%s")

# LSTM neural network training

# res = []
# lstm_models = []

res, lstm_models = get_results("LSTM")

# for LSTM_UNITS in range(3, 5):
#     for HIDDEN_LAYER_UNITS in range(3, 5):
#         res.append([LSTM_UNITS, HIDDEN_LAYER_UNITS])
#
#         train_predictors, train_response, train_conditions = create_dataset(train, WINDOW)
#         test_predictors, test_response, test_conditions = create_dataset(test, WINDOW)
#         train_predictors = train_predictors.transpose(0, 2, 1)
#         test_predictors = test_predictors.transpose(0, 2, 1)
#
#         # model = tf.keras.Sequential()
#         # if CONDITIONAL_RNN:
#         #     model.add(ConditionalRecurrent(layers.LSTM(LSTM_UNITS)))
#         # else:
#         #     model.add(layers.LSTM(LSTM_UNITS))
#         #
#         # if HIDDEN_LAYER_UNITS > 0:
#         #     model.add(layers.Dense(HIDDEN_LAYER_UNITS))
#         # model.add(layers.Dense(11))
#         #
#         # model.compile(loss='mean_squared_error', optimizer='adam')
#
#         model = create_model(LSTM_UNITS, HIDDEN_LAYER_UNITS)
#         model = train_model(model, train_predictors, train_response, train_conditions)
#
#         # model.fit(
#         #     x=[train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors, \
#         #     y=train_response, epochs=500, batch_size=1)
#
#         train_prediction = model([train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors)
#         test_prediction = model([test_predictors, test_conditions] if CONDITIONAL_RNN else test_predictors)
#         res.append(math.sqrt(
#             mean_squared_error(scaler.inverse_transform(train_response), scaler.inverse_transform(train_prediction))))
#         res.append(math.sqrt(
#             mean_squared_error(scaler.inverse_transform(test_response), scaler.inverse_transform(test_prediction))))
#         lstm_models.append(model)

np.savetxt("lstm.txt", np.asarray(res), fmt="%s")

# Best model selection

gru_mod = gru_models[1]
lstm_mod = lstm_models[3]

# Graph comparing predictions and actual values by year

gru_train_prediction = gru_mod(
    [train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors
)
gru_test_prediction = gru_mod(
    [test_predictors, test_conditions] if CONDITIONAL_RNN else test_predictors
)
lstm_train_prediction = lstm_mod(
    [train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors
)
lstm_test_prediction = lstm_mod(
    [test_predictors, test_conditions] if CONDITIONAL_RNN else test_predictors
)


def make_plots(model_to_plot, train_prediction, test_prediction, plot_women):
    train_prediction_plot = scaler.inverse_transform(train_prediction)
    test_prediction_plot = scaler.inverse_transform(test_prediction)
    train_response_plot = scaler.inverse_transform(train_response)
    test_response_plot = scaler.inverse_transform(test_response)

    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    x = np.unique(dataset.index)
    age_groups = [0, 2, 6, 10]
    titles = ["Lithuanian men, ", "Latvian men, ", "Estonian men, "]

    for i in range(3):
        for j in range(4):
            age_group = age_groups[j]

            train_plot = np.empty_like(dataset.iloc[:61, 4:])
            train_plot[:] = np.nan
            ind = int(len(train_prediction_plot) / 6)
            train_plot[WINDOW : ind + WINDOW, :] = train_prediction_plot[
                ind * (2 * i + int(plot_women)) : (ind * (2 * i + 1 + int(plot_women))),
                :,
            ]

            test_plot = np.empty_like(dataset.iloc[:61, 4:])
            test_plot[:] = np.nan
            ts_ind = int(len(test_prediction_plot) / 6)
            test_plot[ind + 2 * WINDOW : len(test_plot), :] = test_prediction_plot[
                ts_ind * (2 * i + int(plot_women)) : (
                    ts_ind * (2 * i + 1 + int(plot_women))
                ),
                :,
            ]

            print(f"dataset: {dataset.head}")
            print(f"dataset length: {len(dataset)}")

            axs[i, j].plot(
                x,
                scaler.inverse_transform(
                    dataset.iloc[
                        61 * (2 * i + int(plot_women)) : 61
                        * (2 * i + 1 + int(plot_women)),
                        4:,
                    ]
                )[:61, age_group],
                label="actual values",
                linewidth=0.7,
                linestyle="dotted",
                color="black",
            )
            axs[i, j].plot(
                x,
                train_plot[:, age_group],
                label="training",
                linewidth=0.7,
                color="forestgreen",
            )
            axs[i, j].plot(
                x,
                test_plot[:, age_group],
                label="forecast",
                linewidth=0.7,
                color="hotpink",
            )
            axs[i, j].set_title(titles[i] + dataset.iloc[4:, 4 + age_group].name)

    for i in range(4):
        axs[2, i].set_xlabel("Year")

    for i in range(3):
        axs[i, 0].set_ylabel("Mortality")
    plt.savefig(model_to_plot + ".png", dpi=300)
    # plt.show()

    # Forecast and actual data by year

    t_train_prediction_plot = train_prediction_plot.transpose(1, 0)
    t_train_response_plot = train_response_plot.transpose(1, 0)
    t_test_prediction_plot = test_prediction_plot.transpose(1, 0)
    t_test_response_plot = test_response_plot.transpose(1, 0)

    t_train_plot = np.empty_like(dataset.iloc[:61, 4:]).transpose(1, 0)
    t_train_plot[:] = np.nan
    t_train_plot[:, : TRAINING_SIZE - WINDOW] = t_train_prediction_plot[
        :, : TRAINING_SIZE - WINDOW
    ]

    t_test_plot = np.empty_like(dataset.iloc[:61, 4:]).transpose(1, 0)
    t_test_plot[:] = np.nan
    t_test_plot[:, TRAINING_SIZE + WINDOW : 62] = t_test_prediction_plot[
        :, : 61 - TRAINING_SIZE - WINDOW
    ]

    plt.plot(
        scaler.inverse_transform(dataset.iloc[:, 4:]).transpose(1, 0)[:, 60],
        label="actual values",
        color="black",
        linewidth=0.7,
    )
    plt.plot(t_test_plot[:, 60], label="forecast", linewidth=0.7, color="red")
    plt.legend(loc="upper left")
    plt.title("Lithuanian male mortality in 2019 (plot of model " + model_to_plot + ")")
    plt.savefig("mort" + model_to_plot + ".png", dpi=300)
    # plt.show()

    plt.plot(
        np.log(scaler.inverse_transform(dataset.iloc[:, 4:]).transpose(1, 0))[:, 60],
        label="actual values",
        color="black",
        linewidth=0.7,
    )
    plt.plot(np.log(t_test_plot)[:, 60], label="forecast", color="red", linewidth=0.7)
    plt.title(
        "Lithuanian male logarithmic mortality in 2019 (plot of model "
        + model_to_plot
        + ")"
    )
    plt.legend(loc="upper left")
    plt.savefig("logmort" + model_to_plot + ".png", dpi=300)
    # plt.show()


make_plots("GRU", gru_train_prediction, gru_test_prediction, False)
make_plots("GRU", gru_train_prediction, gru_test_prediction, True)
make_plots("LSTM", lstm_train_prediction, lstm_test_prediction, False)
make_plots("LSTM", lstm_train_prediction, lstm_test_prediction, True)

# MODEL_FOR_PLOTTING = "GRU"

# if MODEL_FOR_PLOTTING == "GRU":
#     train_prediction_plot = scaler.inverse_transform(gru_train_prediction)
#     test_prediction_plot = scaler.inverse_transform(gru_test_prediction)
# elif MODEL_FOR_PLOTTING == "LSTM":
#     train_prediction_plot = scaler.inverse_transform(lstm_train_prediction)
#     test_prediction_plot = scaler.inverse_transform(lstm_test_prediction)
# train_response_plot = scaler.inverse_transform(train_response)
# test_response_plot = scaler.inverse_transform(test_response)
#
# PLOT_WOMEN = False

# fig, axs = plt.subplots(3, 4, figsize=(20, 15))
# x = np.unique(dataset.index)
# age_groups = [0, 2, 6, 10]
# titles = ["Lithuanian men, ", "Latvian men, ", "Estonian men, "]
#
# for i in range(3):
#     for j in range(4):
#         age_group = age_groups[j]
#
#         train_plot = np.empty_like(dataset.iloc[:61, 4:])
#         train_plot[:] = np.nan
#         ind = int(len(train_prediction_plot) / 6)
#         train_plot[WINDOW:ind + WINDOW, :] = train_prediction_plot[
#                                              ind * (2 * i + int(PLOT_WOMEN)):(ind * (2 * i + 1 + int(PLOT_WOMEN))), :]
#
#         test_plot = np.empty_like(dataset.iloc[:61, 4:])
#         test_plot[:] = np.nan
#         ts_ind = int(len(test_prediction_plot) / 6)
#         test_plot[ind + 2 * WINDOW:len(test_plot), :] = test_prediction_plot[ts_ind * (2 * i + int(PLOT_WOMEN)):(
#                 ts_ind * (2 * i + 1 + int(PLOT_WOMEN))), :]
#
#         axs[i, j].plot(x, scaler.inverse_transform(
#             dataset.iloc[61 * (2 * i + int(PLOT_WOMEN)):61 * (2 * i + 1 + int(PLOT_WOMEN)), 4:])[:61, age_group],
#                        label="actual values", linewidth=0.7, linestyle="dotted", color="black")
#         axs[i, j].plot(x, train_plot[:, age_group], label="training", linewidth=0.7, color="forestgreen")
#         axs[i, j].plot(x, test_plot[:, age_group], label="forecast", linewidth=0.7, color="hotpink")
#         axs[i, j].set_title(titles[i] + dataset.iloc[4:, 4 + age_group].name)
#
# for i in range(4):
#     axs[2, i].set_xlabel("Year")
#
# for i in range(3):
#     axs[i, 0].set_ylabel("Mortality")
# plt.savefig(MODEL_FOR_PLOTTING + ".png", dpi=300)
# plt.show()
#
# # Forecast and actual data by year
#
# t_train_prediction_plot = train_prediction_plot.transpose(1, 0)
# t_train_response_plot = train_response_plot.transpose(1, 0)
# t_test_prediction_plot = test_prediction_plot.transpose(1, 0)
# t_test_response_plot = test_response_plot.transpose(1, 0)
#
# t_train_plot = np.empty_like(dataset.iloc[:61, 4:]).transpose(1, 0)
# t_train_plot[:] = np.nan
# t_train_plot[:, :TRAINING_SIZE - WINDOW] = t_train_prediction_plot[:, :TRAINING_SIZE - WINDOW]
#
# t_test_plot = np.empty_like(dataset.iloc[:61, 4:]).transpose(1, 0)
# t_test_plot[:] = np.nan
# t_test_plot[:, TRAINING_SIZE + WINDOW:62] = t_test_prediction_plot[:, :61 - TRAINING_SIZE - WINDOW]
#
# plt.plot(
#     scaler.inverse_transform(dataset.iloc[:, 4:]).transpose(1, 0)[:, 60],
#     label="actual values",
#     color="black",
#     linewidth=0.7)
# plt.plot(t_test_plot[:, 60], label="forecast", linewidth=0.7, color="red")
# plt.legend(loc="upper left")
# plt.title("Lithuanian male mortality in 2019 (plot of model " + MODEL_FOR_PLOTTING +")")
# plt.savefig("mort" + MODEL_FOR_PLOTTING + ".png", dpi=300)
# plt.show()
#
# plt.plot(
#     np.log(
#         scaler.inverse_transform(dataset.iloc[:, 4:]).transpose(1, 0))[:, 60],
#     label="actual values",
#     color="black",
#     linewidth=0.7)
# plt.plot(np.log(t_test_plot)[:, 60], label="forecast", color="red", linewidth=0.7)
# plt.title("Lithuanian male logarithmic mortality in 2019 (plot of model " + MODEL_FOR_PLOTTING + ")")
# plt.legend(loc="upper left")
# plt.savefig("logmort" + MODEL_FOR_PLOTTING + ".png", dpi=300)
# plt.show()

# Illustrating the change in mortality in 1959-2019

fig, axs = plt.subplots(3, 2, figsize=(14, 10))
plt.subplots_adjust(wspace=0.1, hspace=0.3)
# colors = plt.cm.jet(np.linspace(0, 1, 61))
colors = plt.get_cmap("jet")(np.linspace(0, 1, 61))

for j in range(3):
    for i in range(61):
        axs[j, 0].plot(
            x,
            np.log(scaler.inverse_transform(dataset.iloc[:, 4:]).transpose(1, 0))[
                :, 61 * j * 2 + i
            ],
            color=colors[i],
            linewidth=0.5,
        )
        axs[j, 1].plot(
            x,
            np.log(scaler.inverse_transform(dataset.iloc[:, 4:]).transpose(1, 0))[
                :, 61 * (j * 2 + 1) + i
            ],
            color=colors[i],
            linewidth=0.5,
        )

limits = []
for i in range(3):
    for j in range(2):
        limits.append(axs[i, j].get_ylim())
limits = np.array(*[limits])

y_limit = (min(limits[:, 0]), max(limits[:, 1]))

axs[0, 0].set_ylabel("Logarithmic mortality")
axs[0, 0].set_title("Lithuanian males")
axs[0, 1].set_title("Lithuanian females")

axs[1, 0].set_ylabel("Logarithmic mortality")
axs[1, 0].set_title("Latvian males")
axs[1, 1].set_title("Latvian females")

axs[2, 0].set_ylabel("Logarithmic mortality")
axs[2, 0].set_xlabel("Age")
axs[2, 0].set_title("Estonian males")
axs[2, 1].set_xlabel("Age")
axs[2, 1].set_title("Estonian females")

axs[0, 0].set_ylim = y_limit
axs[0, 1].set_ylim = y_limit
axs[1, 0].set_ylim = y_limit
axs[1, 1].set_ylim = y_limit
axs[2, 0].set_ylim = y_limit
axs[2, 1].set_ylim = y_limit

plt.savefig("alldata.png", dpi=300)
plt.show()

# Illustrating average mortality change in 1959-2019

plt.figure(figsize=(12, 8))
labels = [
    "Lithuanian males",
    "Lithuanian females",
    "Latvian males",
    "Latvian females",
    "Estonian males",
    "Estonian " "females",
]

for j in range(6):
    plt.plot(
        np.unique(dataset.index),
        np.nanmean(
            np.log(scaler.inverse_transform(dataset.iloc[:, 4:]))[
                (61 * j) : 61 * (j + 1), :
            ],
            axis=1,
        ),
        color=colors[j],
        label=labels[j],
    )

plt.ylabel("Logarithmic mortality")
plt.xlabel("Year")
plt.legend(loc="upper right")
plt.savefig("comparison.png", dpi=300)
plt.show()
