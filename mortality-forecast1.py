import datetime
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
# import plotly

import numpy as np
import pandas as pd

import sklearn
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
import tensorflow as tf
# from tensorflow.keras import layers
from keras.layers import RNN, GRU, LSTM, GRUCell, LSTMCell, Dense

# from cond_rnn import ConditionalRecurrent

TRAINING_SIZE = 49 # kiek stebėjimų naudojama prognozavimui
WINDOW = 3 # kokio ilgio praeities reikšmių seka naudojama prognozavimui
CONDITIONAL_RNN = False # ar naudoti lyties ir šalies indikatorius

tf.random.set_seed(1918)

# duomenų nuskaitymas ir paruošimas

def read_data(filenames):
  dataframes = []
  for filename in filenames:
    dataframes.append(
      pd.read_csv(filename, header=1, delim_whitespace=True))
  return dataframes

def clean_data(dataframes):
  clean_dataframes = []
  for dataframe in dataframes:
    dataframe = dataframe.drop(columns="Total")
    dataframe = long_to_wide(dataframe)
    clean_dataframes.append(dataframe)
  return clean_dataframes

def long_to_wide(dataset):
  dataset = dataset.melt(
    id_vars=["Year", "Age"], value_vars=["Male", "Female"])
  dataset = dataset.pivot(
    columns=["Age"], index=["Year", "variable"], values=["value"])
  dataset = dataset.reset_index()
  dataset["variable"] = dataset["variable"] == "Female"
  dataset = dataset.astype('float')
  dataset["Year"] = dataset["Year"].transform(
    lambda x:datetime.date(int(x), 1, 1))
  dataset.columns = dataset.columns.get_level_values(1)
  dataset.columns.values[0] = "Year"
  dataset.columns.values[1] = "F"
  dataset = dataset.set_index(dataset.columns[0])
  dataset = dataset.astype('float')
  dataset = dataset.sort_values(by=['F', 'Year'])
  return dataset

lt_mortality_raw, lt_deaths_raw, lt_exposure_raw, \
lv_mortality_raw, lv_deaths_raw, lv_exposure_raw, \
ee_mortality_raw, ee_deaths_raw, ee_exposure_raw = read_data(
  (
    "./data/LT_Mx_5x1.txt", "./data/LT_Deaths_5x1.txt", 
    "./data/LT_Exposures_5x1.txt", "./data/LV_Mx_5x1.txt",
    "./data/LV_Deaths_5x1.txt", "./data/LV_Exposures_5x1.txt",
    "./data/EE_Mx_5x1.txt", "./data/EE_Deaths_5x1.txt",
    "./data/EE_Exposures_5x1.txt"
  ))

lt_mortality, lv_mortality, ee_mortality = \
  lt_mortality_raw.replace(".", None), lv_mortality_raw.replace(".", None), \
  ee_mortality_raw.replace(".", None)

lt_mortality, lt_deaths, lt_exposure = clean_data(
  (lt_mortality, lt_deaths_raw, lt_exposure_raw))

lv_mortality, lv_deaths, lv_exposure = clean_data(
  (lv_mortality, lv_deaths_raw, lv_exposure_raw))

ee_mortality, ee_deaths, ee_exposure = clean_data(
  (ee_mortality, ee_deaths_raw, ee_exposure_raw))

lt_mortality["LT"], lv_mortality["LV"], ee_mortality["EE"] = 1, 1, 1

lt_deaths, lt_mortality, lt_exposure = \
  lt_deaths.drop(lt_mortality.tail(1).index), \
  lt_mortality.drop(lt_mortality.tail(1).index), \
  lt_exposure.drop(lt_mortality.tail(1).index)

# mirtingumo grupavimas į platesnius intervalus

def sum_groups(dataframe, combined_groups, summing_groups):
  for i in range(len(combined_groups)):
    dataframe[combined_groups[i]] = \
      dataframe.loc[:, summing_groups[i]].sum(axis=1)
  return dataframe

def combine_mortality(mortality_df, deaths_df, exposure_df,
                      combined_groups, summing_groups):
  for i in range(len(combined_groups)):
    mortality_df[combined_groups[i]] = \
     deaths_df[combined_groups[i]] / exposure_df[combined_groups[i]]
    mortality_df = mortality_df.drop(columns=summing_groups[i])
  return mortality_df

combined_groups = ["1-9"]
summing_groups = [["1-4", "5-9"]]

for i in range(1, 9):
  combined_groups.append(str(i)+"0-"+str(i)+"9")
combined_groups.append("90+")

for i in range(1, 9):
  summing_groups.append([str(i)+"0-"+str(i)+"4", str(i)+"5-"+str(i)+"9"])
summing_groups.append(["90-94", "95-99", "100-104", "105-109", "110+"])

combined_groups.insert(0, "0")
summing_groups.insert(0, ["0"])
x = combined_groups

lt_deaths, lt_exposure = \
  sum_groups(lt_deaths, combined_groups, summing_groups), \
  sum_groups(lt_exposure, combined_groups, summing_groups)

lv_deaths, lv_exposure = \
  sum_groups(lv_deaths, combined_groups, summing_groups), \
  sum_groups(lv_exposure, combined_groups, summing_groups)

ee_deaths, ee_exposure = \
  sum_groups(ee_deaths, combined_groups, summing_groups), \
  sum_groups(ee_exposure, combined_groups, summing_groups)

lt_mortality, lv_mortality, ee_mortality = \
  combine_mortality(
    lt_mortality, lt_deaths, lt_exposure, combined_groups, summing_groups), \
  combine_mortality(
    lv_mortality, lv_deaths, lv_exposure, combined_groups, summing_groups), \
  combine_mortality(
    ee_mortality, ee_deaths, ee_exposure, combined_groups, summing_groups)

# duomenų išskyrimas į mokymo ir testavimo imtis

def create_dataset(dataset, length):
  predictors, response, conditions = [], [], []

  for i in range(len(dataset) - length):
    value = dataset.iloc[i:(i + length)]
    if (np.unique(value.iloc[:, 0]).size \
        * np.unique(value.iloc[:, 1]).size \
        * np.unique(value.iloc[:, 2]).size \
        * np.unique(value.iloc[:, 3]).size == 1) and \
        (dataset.iloc[i:i+length+1, :].index.is_monotonic_increasing):
      predictors.append(value.iloc[:, 4:])
      response.append(dataset.iloc[i + length, 4:])
      conditions.append(
        [value.iloc[0, 0], value.iloc[0, 1], value.iloc[0, 2], \
        value.iloc[0, 3]])

  return np.array(predictors), np.array(response), np.array(conditions)

dataset = pd.concat([lt_mortality, lv_mortality, ee_mortality])

dataset[["LT", "LV", "EE"]] = dataset[["LT", "LV", "EE"]].fillna(0)

dataset.insert(0, "LT", dataset.pop("LT"))
dataset.insert(1, "LV", dataset.pop("LV"))
dataset.insert(2, "EE", dataset.pop("EE"))

scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
dataset.iloc[:, 4:] = scaler.fit_transform(dataset.iloc[:, 4:])

train = dataset[np.mod(np.arange(dataset.shape[0]),61)<TRAINING_SIZE]
test = pd.concat([train, dataset]).drop_duplicates(keep=False)

# GRU neuroninių tinklų apmokymas

res = []
gru_models = []

for i in range(3, 5):
  for j in range(3, 5):
    res.append([i, j])
    GRU_UNITS = i
    HIDDEN_LAYER_UNITS = j

    train_predictors, train_response, train_conditions = create_dataset(
      train, WINDOW)
    test_predictors, test_response, test_conditions = create_dataset(
      test, WINDOW)
    train_predictors = train_predictors.transpose(0, 2, 1)
    test_predictors = test_predictors.transpose(0, 2, 1)

    model = tf.keras.Sequential()
    if CONDITIONAL_RNN:
      model.add(RNN(GRUCell(GRU_UNITS)))
    else:
      model.add(GRU(GRU_UNITS))

    if HIDDEN_LAYER_UNITS > 0:
      model.add(Dense(HIDDEN_LAYER_UNITS))
    model.add(Dense(11))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(
    x = [train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors, \
    y = train_response, epochs=500, batch_size=1)

    train_prediction = model([train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors)
    test_prediction = model([test_predictors, test_conditions] if CONDITIONAL_RNN else test_predictors)
    res.append(math.sqrt(mean_squared_error(scaler.inverse_transform(train_response), scaler.inverse_transform(train_prediction))))
    res.append(math.sqrt(mean_squared_error(scaler.inverse_transform(test_response), scaler.inverse_transform(test_prediction))))
    gru_models.append(model)
        
np.savetxt('gru.txt', np.asarray(res), fmt='%s')

# LSTM neuroninių tinklų apmokymas

res = []
lstm_models = []

for i in range(3, 5):
  for j in range(3, 5):
    res.append([i, j])
    LSTM_UNITS = i
    HIDDEN_LAYER_UNITS = j

    train_predictors, train_response, train_conditions = create_dataset(train, WINDOW)
    test_predictors, test_response, test_conditions = create_dataset(test, WINDOW)
    train_predictors = train_predictors.transpose(0, 2, 1)
    test_predictors = test_predictors.transpose(0, 2, 1)

    model = tf.keras.Sequential()
    if CONDITIONAL_RNN:
      model.add(RNN(LSTMCell(LSTM_UNITS)))
    else:
      model.add(LSTM(LSTM_UNITS))

    if HIDDEN_LAYER_UNITS > 0:
      model.add(Dense(HIDDEN_LAYER_UNITS))
    model.add(Dense(11))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(
    x = [train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors, \
    y = train_response, epochs=500, batch_size=1)

    train_prediction = model([train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors)
    test_prediction = model([test_predictors, test_conditions] if CONDITIONAL_RNN else test_predictors)
    res.append(math.sqrt(mean_squared_error(scaler.inverse_transform(train_response), scaler.inverse_transform(train_prediction))))
    res.append(math.sqrt(mean_squared_error(scaler.inverse_transform(test_response), scaler.inverse_transform(test_prediction))))
    lstm_models.append(model)

np.savetxt('lstm.txt', np.asarray(res), fmt='%s')

# geriausi modeliai

gru_mod = gru_models[1]
lstm_mod = lstm_models[3]

# grafikas palyginti prognozes ir tikrąsias reikšmes pagal metus

gru_train_prediction = gru_mod([train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors)
gru_test_prediction = gru_mod([test_predictors, test_conditions] if CONDITIONAL_RNN else test_predictors)
lstm_train_prediction = lstm_mod([train_predictors, train_conditions] if CONDITIONAL_RNN else train_predictors)
lstm_test_prediction = lstm_mod([test_predictors, test_conditions] if CONDITIONAL_RNN else test_predictors)

MODEL_FOR_PLOTTING = "GRU"

if MODEL_FOR_PLOTTING == "GRU":
  train_prediction_plot = scaler.inverse_transform(gru_train_prediction)
  test_prediction_plot = scaler.inverse_transform(gru_test_prediction)
elif MODEL_FOR_PLOTTING == "LSTM":
  train_prediction_plot = scaler.inverse_transform(lstm_train_prediction)
  test_prediction_plot = scaler.inverse_transform(lstm_test_prediction)
train_response_plot = scaler.inverse_transform(train_response)
test_response_plot = scaler.inverse_transform(test_response)

PLOT_WOMEN = False

fig, axs = plt.subplots(3, 4, figsize=(20,15))
x = np.unique(dataset.index)
agegroups = [0, 2, 6, 10]
titles = ["Lietuvos vyrai, ", "Latvijos vyrai, ", "Estijos vyrai, "]

for i in range(3):
  for j in range(4):
    agegroup = agegroups[j]
    
    train_plot = np.empty_like(dataset.iloc[:61, 4:])
    train_plot[:] = np.nan
    ind = int(len(train_prediction_plot) / 6)
    train_plot[WINDOW:ind+WINDOW, :] = train_prediction_plot[ind*(2*i+int(PLOT_WOMEN)):(ind*(2*i+1+int(PLOT_WOMEN))), :]
    
    test_plot = np.empty_like(dataset.iloc[:61, 4:])
    test_plot[:] = np.nan
    ts_ind = int(len(test_prediction_plot) / 6)
    test_plot[ind+2*WINDOW:len(test_plot), :] = test_prediction_plot[ts_ind*(2*i+int(PLOT_WOMEN)):(ts_ind*(2*i+1+int(PLOT_WOMEN))), :]
    
    axs[i, j].plot(x, scaler.inverse_transform(
      dataset.iloc[61*(2*i+int(PLOT_WOMEN)):61*(2*i+1+int(PLOT_WOMEN)), 4:])[:61, agegroup], label="tikros reikšmės", linewidth=0.7, linestyle="dotted", color="black")
    axs[i, j].plot(x, train_plot[:, agegroup], label="mokymasis", linewidth=0.7, color="forestgreen")
    axs[i, j].plot(x, test_plot[:, agegroup], label="prognozė", linewidth=0.7, color="hotpink")
    axs[i, j].set_title(titles[i]+dataset.iloc[4:, 4+agegroup].name)

for i in range(4):
  axs[2, i].set_xlabel("Metai")
  
for i in range(3):
  axs[i, 0].set_ylabel("Mirtingumas")
plt.savefig(MODEL_FOR_PLOTTING + ".png", dpi=300)    
plt.show()

# grafikas palyginti prognozes ir tikrąsias reikšmes pagal amžių

t_train_prediction_plot = train_prediction_plot.transpose(1, 0)
t_train_response_plot = train_response_plot.transpose(1, 0)
t_test_prediction_plot = test_prediction_plot.transpose(1, 0)
t_test_response_plot = test_response_plot.transpose(1, 0)

t_train_plot = np.empty_like(dataset.iloc[:61, 4:]).transpose(1, 0)
t_train_plot[:] = np.nan
t_train_plot[:, :TRAINING_SIZE-WINDOW] = t_train_prediction_plot[:, :TRAINING_SIZE-WINDOW]

t_test_plot = np.empty_like(dataset.iloc[:61, 4:]).transpose(1, 0)
t_test_plot[:] = np.nan
t_test_plot[:, TRAINING_SIZE+WINDOW:62] = t_test_prediction_plot[:, :61-TRAINING_SIZE-WINDOW]

plt.plot(
  scaler.inverse_transform(dataset.iloc[:, 4:]).transpose(1, 0)[:, 60], \
  label="tikros reikšmės", \
  color="black", \
  linewidth=0.7)
plt.plot(t_test_plot[:, 60], label="prognozė", linewidth=0.7, color="red")
plt.legend(loc="upper left")
plt.title("Lietuvos vyrų mirtingumas 2019 m. (" + MODEL_FOR_PLOTTING + " modelio prognozė)")
plt.savefig("mort" + MODEL_FOR_PLOTTING + ".png", dpi=300)
plt.show()

plt.plot(
  np.log(
    scaler.inverse_transform(dataset.iloc[:, 4:]).transpose(1, 0))[:, 60], \
  label="tikros reikšmės", \
  color="black", \
  linewidth=0.7)
plt.plot(np.log(t_test_plot)[:, 60], label="prognozė", color="red", linewidth=0.7)
plt.title("Lietuvos vyrų logaritminis mirtingumas 2019 m. (" + MODEL_FOR_PLOTTING + " modelio prognozė)")
plt.legend(loc="upper left")
plt.savefig("logmort" + MODEL_FOR_PLOTTING + ".png", dpi=300)
plt.show()

# grafikas iliustruoti mirtingumo kaitą 1959-2019 m.

fig, axs = plt.subplots(3, 2, figsize=(14, 10))
plt.subplots_adjust(wspace=0.1,
                    hspace=0.3)
colors = plt.cm.jet(np.linspace(0,1,61))

for j in range(3):
  for i in range(61):
    axs[j, 0].plot(x, np.log(scaler.inverse_transform(dataset.iloc[:, 4:]).transpose(1, 0))[:, 61 * j * 2 + i], color=colors[i], linewidth=0.5)
    axs[j, 1].plot(x, np.log(scaler.inverse_transform(dataset.iloc[:, 4:]).transpose(1, 0))[:, 61 * (j * 2 + 1) + i], color=colors[i], linewidth=0.5)

limits = []
for i in range(3):
  for j in range(2):
    limits.append(axs[i, j].get_ylim())
limits = np.array(*[limits])
    
y_limit = (min(limits[:, 0]), max(limits[:, 1]))
    
axs[0, 0].set_ylabel("Logaritminis mirtingumas")
axs[0, 0].set_title("Lietuvos vyrai")
axs[0, 1].set_title("Lietuvos moterys")

axs[1, 0].set_ylabel("Logaritminis mirtingumas")
axs[1, 0].set_title("Latvijos vyrai")
axs[1, 1].set_title("Latvijos moterys")

axs[2, 0].set_ylabel("Logaritminis mirtingumas")
axs[2, 0].set_xlabel("Amžius")
axs[2, 0].set_title("Estijos vyrai")
axs[2, 1].set_xlabel("Amžius")
axs[2, 1].set_title("Estijos moterys")

axs[0, 0].set_ylim = y_limit
axs[0, 1].set_ylim = y_limit
axs[1, 0].set_ylim = y_limit
axs[1, 1].set_ylim = y_limit
axs[2, 0].set_ylim = y_limit
axs[2, 1].set_ylim = y_limit

plt.savefig("alldata.png", dpi=300)
plt.show()

# grafikas iliustruoti vidutinio mirtingumo kaitą 1959-2019 m.

plt.figure(figsize=(12, 8))
labels = ["Lietuvos vyrai", "Lietuvos moterys", "Latvijos vyrai",
          "Latvijos moterys", "Estijos vyrai", "Estijos moterys"]

for j in range(6):
  plt.plot(np.unique(dataset.index), np.nanmean(np.log(scaler.inverse_transform(dataset.iloc[:, 4:]))[(61 * j):61 * (j + 1), :], axis=1), color=plt.cm.Paired(j), label=labels[j])
  
plt.ylabel("Logaritminis mirtingumas")
plt.xlabel("Metai")
plt.legend(loc="upper right")
plt.savefig("comparison.png", dpi=300)
plt.show()
