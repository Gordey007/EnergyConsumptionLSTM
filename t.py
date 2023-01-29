# univariate bidirectional lstm example
import pandas as pd
from keras.metrics import RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError, Accuracy
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
# Графические пакеты
import matplotlib.pyplot as plt

# WebAgg/Qt5Agg
plt.matplotlib.use('WebAgg')


# split a univariate sequence
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
# raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_ahead = 72
n_days = 1
start_data = 0

raw_seq = pd.read_csv('input\openweathermap_komsomolsk_on_amur_2001_2022.csv')
d = raw_seq[start_data:len(raw_seq) - n_ahead * n_days]['dew_point_degrees'].tolist()
true_value = raw_seq[len(raw_seq) - n_ahead * n_days:]['dew_point_degrees'].tolist()

# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(d, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',
              metrics=[RootMeanSquaredError(),
                       MeanSquaredError(),
                       MeanAbsoluteError(),
                       Accuracy()]
              )
# fit model
model.fit(X, y, epochs=2, verbose=1)
# # demonstrate prediction
# x_input = array([70, 80, 90])
# x_input = x_input.reshape((1, n_steps, n_features))
# result = model.predict(x_input, verbose=0)
# # print(yhat)
#
#
# # Absolute error
# a = []
# for i in range(0, (n_ahead * n_days)):
#     a.append(round(abs(true_value[i] - result[i]), 2))
#
# absolute_error = round(sum(a) / (n_ahead * n_days), 2)
#
# # Relative errors
# e = []
# for i in range(0, (n_ahead * n_days)):
#     e.append(round(abs(true_value[i] - result[i]) / true_value[i] * 100, 2))
#
# relative_error = round(abs(sum(e) / (n_ahead * n_days)))
#
# # Составление прогнозов (график)
# plt.figure(figsize=(15, 6))
# plt.plot(true_value, c="blue", linewidth=3, label='Реальные значения',
#          marker='o')
#
# for x, y in zip(true_value[0:len(true_value):10],
#                 range(0, (n_ahead * n_days))[0:n_ahead * n_days:10]):
#     plt.text(y + .8, x + .2, str(x), color='blue')
#
# plt.plot(result, c="green", linewidth=3, label='Прогнозируемые значения', marker='X')
# for x, y in zip(result[0:len(result):10], range(0, (n_ahead * n_days))[0:n_ahead * n_days:10]):
#     plt.text(y + .8, x + .2, str(x), color='green')
#
# plt.title('Модель №7 - ConvLSTM2D')
# plt.xticks(range(0, (n_ahead * n_days)))
# plt.xlabel(f'Часы\n'
#            f'Абсолютная ошибка: {absolute_error}, относительная ошибка: {relative_error}%')
# plt.ylabel('Погода')
#
# plt.grid()
# plt.legend()
#
# plt.show()
