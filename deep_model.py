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


class DeepModelTS:
    def __init__(
            self,
            data: pd.DataFrame,
            Y_var: str,
            lag: int,
            LSTM_layer_depth: int,
            epochs=10,
            batch_size=256,
            train_test_split=0,
    ):

        self.model = None
        self.data = data
        self.Y_var = Y_var
        self.lag = lag
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split

    # Метод создания матрицы X и Y из списка временных рядов
    # для обучения модели глубокого обучения
    @staticmethod
    def create_X_Y(ts: list, lag: int) -> tuple:
        X, Y = [], []

        if len(ts) - lag <= 0:
            X.append(ts)
        else:
            for i in range(len(ts) - lag):
                Y.append(ts[i + lag])
                X.append(ts[i:(i + lag)])

        X, Y = np.array(X), np.array(Y)

        # Преобразование массива X в форму ввода LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, Y

    # Метод создания данных для модели нейронной сети
    def create_data_for_NN(self, use_last_n=None):
        # Извлечение основной переменной, которую хотим прогнозировать
        # y = list(map(lambda x: round(x - 273.15, 2), self.data[self.Y_var]))
        y = round(self.data[self.Y_var], 2).tolist()

        # Подстановка временных рядов при необходимости
        if use_last_n is not None:
            y = y[-use_last_n:]

        # Матрица X будет содержать N (например, lag = 3) измерений назад от значения Y
        # Если Y = [1434.0, 1489.0, 1620.0], то
        # X = [
        #     [1621.0, 1536.0, 1500.0],
        #     [1536.0, 1500.0, 1434.0],
        #     [1500.0, 1434.0, 1489.0],
        # ]
        # Первые три значения до 1434,0 (1500.0, 1536.0, 1621.0, 1434.0)
        X, Y = self.create_X_Y(y, self.lag)

        # Создание обучающих и тестовых наборов
        X_train = X
        X_test = []

        Y_train = Y
        Y_test = []

        if self.train_test_split > 0:
            index = round(len(X) * self.train_test_split)

            X_train = X[:(len(X) - index)]
            X_test = X[-index:]

            Y_train = Y[:(len(X) - index)]
            Y_test = Y[-index:]

        return X_train, X_test, Y_train, Y_test

    # Метод создания модели LSTM
    def LSTModel(self):
        # Получение данных
        X_train, X_test, Y_train, Y_test = self.create_data_for_NN()
        # Определение модели
        model = Sequential()

        X_train = X_train.reshape((X_train.shape[0], 1, 1, self.lag, 1))

        # model.add(ConvLSTM2D(filters=256,
        #                      kernel_size=(5, 5),
        #                      padding='same',
        #                      activation='relu',
        #                      input_shape=(1, 1, self.lag, 1), return_sequences=True))
        # model.add(ConvLSTM2D(filters=256,
        #                      kernel_size=(3, 3),
        #                      padding='same',
        #                      activation='relu',
        #                      input_shape=(1, 1, self.lag, 1), return_sequences=True))
        model.add(ConvLSTM2D(filters=256,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu',
                             input_shape=(1, 1, self.lag, 1)))
        model.add(Flatten())
        model.add(Dense(100))
        # BatchNormalization(
        #         momentum=0.95,
        #         epsilon=0.005,
        #         beta_initializer=RandomNormal(mean=0.0, stddev=0.05),
        #         gamma_initializer=Constant(value=0.9)
        # )
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

        # Определение параметра модели
        keras_dict = {
            'x': X_train,
            'y': Y_train,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'shuffle': False
        }

        # if self.train_test_split > 0:
        #     keras_dict.update({
        #         'validation_data': (X_test, Y_test)
        #     })

        # Обучение модели
        model.fit(**keras_dict)
        # model.save('my_model.h5')
        # model = load_model('my_model.h5', compile=False)

        # Сохранение модели в классе
        self.model = model

        return model

    # Метод прогнозирования с использованием тестовых данных,
    # использованных при создании класса
    def predict(self) -> list:
        yhat = []

        if self.train_test_split > 0:
            # Получение последних n временных рядов
            _, X_test, _, _ = self.create_data_for_NN()

            # Составление списка предсказаний
            yhat = [y[0] for y in self.model.predict(X_test)]

        return yhat

    # Метод прогнозирования N временных шагов вперед
    def predict_n_ahead(self, n_ahead: int):
        X, _, _, _ = self.create_data_for_NN(use_last_n=self.lag)

        # Составление списка предсказаний
        yhat = []

        for _ in range(n_ahead):
            # Прогноз
            X = X.reshape((1, 2, 1, self.lag, 1))

            fc = self.model.predict(X)
            yhat.append(fc)

            # Создание новой входной матрицы для прогнозирования
            X = np.append(X, fc)
            # Исключение первой переменной
            X = np.delete(X, 0)
            # Изменение формы для следующей итерации
            X = np.reshape(X, (1, len(X), 1))

        return yhat

    # Для обеспечения воспроизводимости результатов на CPU
    # tf.random.set_seed(13)
    print(tf.config.list_physical_devices('GPU'))
