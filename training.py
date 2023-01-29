# Обработка данных
import pandas as pd
import numpy as np

# Глубокое обучение
from keras.initializers.initializers_v2 import RandomNormal, Constant
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.metrics import RootMeanSquaredError, Accuracy
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Conv1D, MaxPooling1D, \
    Flatten, ConvLSTM2D, RepeatVector, BatchNormalization
import tensorflow as tf

# Класс для создания модели глубокого обучения с использованием временного ряда
from keras.optimizers import Adam
from keras.saving.save import load_model
from numpy import array
import pickle

import utils

# Обработка данных
import pandas as pd
# Чтение файла конфигурации
import yaml
# Управление каталогом
import os
# Графические пакеты
import matplotlib.pyplot as plt

# WebAgg/Qt5Agg
plt.matplotlib.use('WebAgg')
# Работа с датой и временим
import time
from datetime import datetime

print(tf.config.list_physical_devices('GPU'))

# Чтение параметров pipeline
with open(f'{os.getcwd()}\\conf.yml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

# Определение модели
model = Sequential()
model.add(ConvLSTM2D(filters=256,
                     kernel_size=(1, 2),
                     padding='same',
                     activation='relu',
                     input_shape=(1, 1, conf.get('lag'), 1), return_sequences=True))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(1))
# Adding the output layer
model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.001),
              metrics=[RootMeanSquaredError(),
                       MeanSquaredError(),
                       MeanAbsoluteError(),
                       Accuracy()]
              )
# вывод информации
model.summary()

templite_date = '%Y-%m-%d %H:%M:%S'

# files = ['AEP_hourly', 'COMED_hourly', 'DAYTON_hourly', 'DEOK_hourly', 'DOM_hourly',
#          'DUQ_hourly', 'EKPC_hourly', 'FE_hourly', 'NI_hourly', 'PJME_hourly', 'PJMW_hourly']

files = ['openweathermap_komsomolsk_on_amur_2001_2022']
i = 1
for file in files:
    print(f'\n#{i}/{len(files)} - {file}\n')
    i = i + 1

    # Чтение данных
    d = pd.read_csv(f'input\\{file}.csv')
    d[conf.get('X_var')] = [datetime.strptime(x, templite_date) for x in
                            d[conf.get('X_var')]]

    columns = ['dew_point_degrees', 'feels_like_degrees', 'temp_min_degrees',
               'temp_max_degrees']

    # columns = ['dew_point_kelvins', 'feels_like_kelvins', 'temp_min_kelvins',
    #            'temp_max_kelvins']

    j = 1
    for colum in columns[:1]:
        print(f'\n#{j}/{len(columns)} - {colum}\n')
        j = j + 1

        # Получение данных
        X_train, X_test, Y_train, Y_test = utils.create_data_for_NN(
                data=d,
                Y_var=colum,
                lag=conf.get('lag')
        )

        X_train = X_train.reshape((X_train.shape[0], 1, 1, conf.get('lag'), 1))

        # Определение параметра модели
        keras_dict = {
            'x': X_train,
            'y': Y_train,
            'batch_size': conf.get('batch_size'),
            'epochs': conf.get('epochs'),
            'shuffle': False
        }

        # Обучение модели
        model.fit(**keras_dict)
        model.save('my_model.h5')

        # model = load_model('my_model.h5', compile=False)
